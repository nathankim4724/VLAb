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

import numpy as np
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.compute_stats import aggregate_feature_stats
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
# from lerobot.envs.configs import EnvConfig  # Removed - not needed for SmolVLA2 pretraining
# from lerobot.envs.utils import env_to_policy_features  # Removed - not needed for SmolVLA2 pretraining
# SmolVLA2-only policy factory - removed all other policies
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla2.configuration_smolvla2 import SmolVLA2Config


def get_policy_class(name: str) -> PreTrainedPolicy:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name == "smolvla2":
        from lerobot.policies.smolvla2.modeling_smolvla2 import SmolVLA2Policy
        return SmolVLA2Policy
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented. Only SmolVLA2 is supported.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "smolvla2":
        return SmolVLA2Config(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available. Only SmolVLA2 is supported.")


def combine_robot_type_stats(stats_by_robot_type: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """Merge per-robot-type stats into one set, count-weighted, independent of dict/set order.

    Per-robot-type stats have every entry (including `count`) padded to max_action_dim/max_state_dim by
    MultiLeRobotDatasetMeta, so only the first `count` entry is the real frame count.
    """
    robot_types = sorted(stats_by_robot_type)
    features = sorted({f for rt in robot_types for f in stats_by_robot_type[rt]})
    combined = {}
    for feature in features:
        per_type = []
        for rt in robot_types:
            if feature not in stats_by_robot_type[rt]:
                continue
            ft = {k: np.asarray(v) for k, v in stats_by_robot_type[rt][feature].items()}
            ft["count"] = ft["count"].reshape(-1)[:1]
            per_type.append(ft)
        combined[feature] = aggregate_feature_stats(per_type)
    return combined


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
) -> PreTrainedPolicy:
    """Make an instance of a policy class.

    This function exists because (for now) we need to parse features from a dataset
    in order to properly dimension and instantiate a policy for that dataset.

    Args:
        cfg (PreTrainedConfig): The config of the policy to make. If `pretrained_path` is set, the policy will
            be loaded with the weights from that path.
        ds_meta (LeRobotDatasetMetadata | None, optional): Dataset metadata to take input/output shapes and
            statistics to use for (un)normalization of inputs/outputs in the policy. Defaults to None.

    Raises:
        ValueError: ds_meta must be provided.

    Returns:
        PreTrainedPolicy: SmolVLA2 policy instance
    """
    if not ds_meta:
        raise ValueError("Dataset metadata must be provided for SmolVLA2 pretraining.")

    # SmolVLA2-only factory - no backend compatibility checks needed

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    # Parse features from dataset metadata (required for SmolVLA2)
    features = dataset_to_policy_features(ds_meta.features)
    # Handle robot-type grouped stats - flatten to feature-level stats  
    if ds_meta.stats and len(ds_meta.stats) == 1:
        # Single robot type - use its stats directly
        robot_type = list(ds_meta.stats.keys())[0]
        kwargs["dataset_stats"] = ds_meta.stats[robot_type]
    elif ds_meta.stats and len(ds_meta.stats) > 1:
        # Multiple robot types - combine their stats, weighted by frame count. Upstream kept, for each
        # feature, the stats of whichever robot type came first in set order, which also changes between
        # processes (and so between resumed jobs, since norm stats are not restored from checkpoints).
        kwargs["dataset_stats"] = combine_robot_type_stats(ds_meta.stats)
    else:
        kwargs["dataset_stats"] = ds_meta.stats

    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy
