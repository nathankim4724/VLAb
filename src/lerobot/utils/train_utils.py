#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import shutil
from pathlib import Path

from termcolor import colored
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.configs.train import TrainPipelineConfig
from lerobot.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from lerobot.datasets.utils import load_json, write_json
from lerobot.optim.optimizers import load_optimizer_state, save_optimizer_state
from lerobot.optim.schedulers import load_scheduler_state, save_scheduler_state
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.random_utils import load_rng_state, save_rng_state


def log_output_dir(out_dir):
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {out_dir}")


def get_step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns the checkpoint sub-directory corresponding to the step number."""
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def save_training_step(step: int, save_dir: Path) -> None:
    write_json({"step": step}, save_dir / TRAINING_STEP)


def load_training_step(save_dir: Path) -> int:
    training_step = load_json(save_dir / TRAINING_STEP)
    return training_step["step"]


# Prefix of in-progress checkpoint writes/deletions inside the checkpoints dir. Step directories only ever
# appear complete: they are written under this prefix and renamed into place.
TMP_CHECKPOINT_PREFIX = ".tmp_"


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    """Points the `last` symlink at `checkpoint_dir`. Call from the main process only.

    The new link is created under a temporary name and renamed over the old one (atomic), so `last` always
    points at a complete checkpoint, even if the job is killed (e.g. preempted) during the update.
    """
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    tmp_link = checkpoint_dir.parent / f"{TMP_CHECKPOINT_PREFIX}{LAST_CHECKPOINT_LINK}"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    tmp_link.symlink_to(checkpoint_dir.relative_to(checkpoint_dir.parent))
    os.replace(tmp_link, last_checkpoint_dir)
    return last_checkpoint_dir


def prune_checkpoints(checkpoints_dir: Path, keep_last: int, keep_every: int | None = None) -> list[Path]:
    """Deletes step checkpoints except the newest `keep_last`, those at steps divisible by `keep_every` (if
    set), and the one `last` points to. Also removes leftovers of interrupted saves and deletions.
    Call from the main process only, after `update_last_checkpoint`.
    """
    if keep_last < 1:
        raise ValueError(f"{keep_last=} must be at least 1.")
    for leftover in checkpoints_dir.glob(f"{TMP_CHECKPOINT_PREFIX}*"):
        if leftover.is_dir() and not leftover.is_symlink():
            shutil.rmtree(leftover)

    step_dirs = sorted(
        (p for p in checkpoints_dir.iterdir() if p.name.isdigit() and p.is_dir() and not p.is_symlink()),
        key=lambda p: int(p.name),
    )
    last_link = checkpoints_dir / LAST_CHECKPOINT_LINK
    last_target = last_link.resolve() if last_link.is_symlink() else None
    removed = []
    for step_dir in step_dirs[:-keep_last]:
        if keep_every and int(step_dir.name) % keep_every == 0:
            continue
        if step_dir.resolve() == last_target:
            continue
        # Rename first so an interrupted deletion never leaves a partial directory under a step name.
        trash = checkpoints_dir / f"{TMP_CHECKPOINT_PREFIX}delete_{step_dir.name}"
        step_dir.rename(trash)
        shutil.rmtree(trash)
        removed.append(step_dir)
    return removed


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
) -> None:
    """This function creates the following directory structure:

    005000/  #  training step at checkpoint
    ├── pretrained_model/
    │   ├── config.json  # policy config
    │   ├── model.safetensors  # policy weights
    │   └── train_config.json  # train config
    └── training_state/
        ├── optimizer_param_groups.json  #  optimizer param groups
        ├── optimizer_state.safetensors  # optimizer state
        ├── rng_state.safetensors  # rng states
        ├── scheduler_state.json  # scheduler state
        └── training_step.json  # training step

    Args:
        cfg (TrainPipelineConfig): The training config used for this run.
        step (int): The training step at that checkpoint.
        policy (PreTrainedPolicy): The policy to save.
        optimizer (Optimizer | None, optional): The optimizer to save the state from. Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler to save the state from. Defaults to None.

    The checkpoint is written to a temporary directory and renamed to `checkpoint_dir` once complete, so a
    job killed mid-save leaves only a `.tmp_*` directory (removed by `prune_checkpoints`). Call from the main
    process only.
    """
    tmp_dir = checkpoint_dir.parent / f"{TMP_CHECKPOINT_PREFIX}{checkpoint_dir.name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    pretrained_dir = tmp_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir)
    cfg.save_pretrained(pretrained_dir)
    save_training_state(tmp_dir, step, optimizer, scheduler)
    if checkpoint_dir.exists():
        # Only when re-saving a step, e.g. after resuming from an older checkpoint.
        shutil.rmtree(checkpoint_dir)
    tmp_dir.rename(checkpoint_dir)


def save_training_state(
    checkpoint_dir: Path,
    train_step: int,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> None:
    """
    Saves the training step, optimizer state, scheduler state, and rng state.

    Args:
        save_dir (Path): The directory to save artifacts to.
        train_step (int): Current training step.
        optimizer (Optimizer | None, optional): The optimizer from which to save the state_dict.
            Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler from which to save the state_dict.
            Defaults to None.
    """
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(train_step, save_dir)
    save_rng_state(save_dir)
    if optimizer is not None:
        save_optimizer_state(optimizer, save_dir)
    if scheduler is not None:
        save_scheduler_state(scheduler, save_dir)


def load_training_state(
    checkpoint_dir: Path, optimizer: Optimizer, scheduler: LRScheduler | None
) -> tuple[int, Optimizer, LRScheduler | None]:
    """
    Loads the training step, optimizer state, scheduler state, and rng state.
    This is used to resume a training run.

    Args:
        checkpoint_dir (Path): The checkpoint directory. Should contain a 'training_state' dir.
        optimizer (Optimizer): The optimizer to load the state_dict to.
        scheduler (LRScheduler | None): The scheduler to load the state_dict to (can be None).

    Raises:
        NotADirectoryError: If 'checkpoint_dir' doesn't contain a 'training_state' dir

    Returns:
        tuple[int, Optimizer, LRScheduler | None]: training step, optimizer and scheduler with their
            state_dict loaded.
    """
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)

    load_rng_state(training_state_dir)
    step = load_training_step(training_state_dir)
    optimizer = load_optimizer_state(optimizer, training_state_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, training_state_dir)

    return step, optimizer, scheduler
