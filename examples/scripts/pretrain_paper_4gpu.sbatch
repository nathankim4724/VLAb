#!/bin/bash
# SmolVLA action-expert pretraining with the paper recipe (arXiv 2506.01844, sec. 4.3): frozen SmolVLM2
# (first 16 layers), randomly initialised action expert, 4 GPUs x 64 = global batch 256, 200k steps,
# AdamW lr 1e-4 with 100 warmup steps and cosine decay to 2.5e-6, bf16, 512x512 images.
#
# Preemption-safe on the free-cycles QoS. Preemption requeues the job (PreemptMode=REQUEUE, only for jobs
# submitted with --requeue) with ~10 s between SIGTERM and SIGKILL, too short to save, so progress since
# the last checkpoint is lost. Every (re)start resumes from the newest complete checkpoint in the run dir.
# Near the 2-day time limit the job requeues itself, because a job that hits its time limit is not requeued.
#
# The first start snapshots the VLAb code (git HEAD, which must be clean) into the run dir, and every
# restart runs that snapshot, so later edits or pulls in the VLAb clone don't change a running baseline.
#
# Full run:
#   sbatch examples/scripts/pretrain_paper_4gpu.sbatch
# 4-GPU smoke test (2 datasets, debug QoS):
#   sbatch --qos=vision-pulkitag-debug --time=02:00:00 \
#     --export=ALL,RUN_NAME=smoke4_001,DATASETS=smoke,STEPS=300,SAVE_FREQ=100,LOG_FREQ=20 \
#     examples/scripts/pretrain_paper_4gpu.sbatch
#
#SBATCH --job-name=smolvla_pretrain
#SBATCH --account=vision-pulkitag-urops
#SBATCH --partition=vision-pulkitag-h200
#SBATCH --qos=vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600
#SBATCH --output=/data/pulkitag/results/nkim/vlab_runs/logs/%x_%j.out

set -uo pipefail

RUN_NAME=${RUN_NAME:-paper_base_4gpu_001}
DATASETS=${DATASETS:-all}          # all: examples/all_datasets_relative.txt | smoke: the 2 smoke-test datasets
STEPS=${STEPS:-200000}
WARMUP_STEPS=${WARMUP_STEPS:-100}
DECAY_STEPS=${DECAY_STEPS:-200000}
BATCH_SIZE=${BATCH_SIZE:-64}       # per GPU
SAVE_FREQ=${SAVE_FREQ:-2000}
KEEP_LAST=${KEEP_LAST:-3}          # newest checkpoints kept ...
KEEP_EVERY=${KEEP_EVERY:-20000}    # ... plus every KEEP_EVERY steps
LOG_FREQ=${LOG_FREQ:-100}
NUM_WORKERS=${NUM_WORKERS:-16}     # per GPU process
WANDB=${WANDB:-true}

VLAB=/data/pulkitag/misc/nkim/VLAb
RUN_DIR=/data/pulkitag/results/nkim/vlab_runs/$RUN_NAME
OUTPUT_DIR=$RUN_DIR/train          # train.py's output_dir: checkpoints, wandb files
CODE_DIR=$RUN_DIR/code             # snapshot of the VLAb code this run uses
CKPTS=$OUTPUT_DIR/checkpoints

log() { echo "[$(date '+%F %T')] $*"; }

log "job ${SLURM_JOB_ID} restart_count=${SLURM_RESTART_COUNT:-0} node=$(hostname) run=$RUN_NAME"
log "gpus: ${CUDA_VISIBLE_DEVICES:-none} (${SLURM_GPUS_ON_NODE:-?} on node), cpus: ${SLURM_CPUS_PER_TASK:-?}"

# ---- Code snapshot (first start only) -------------------------------------------------------------------
# Upstream VLAb tracks a few __pycache__/*.pyc files, which any run from the clone rewrites: ignore them.
CODE_PATHS=(src assets accelerate_configs examples ':(exclude)*.pyc')
if [[ ! -f $CODE_DIR/COMMIT ]]; then
  if [[ -n $(git -C "$VLAB" status --porcelain -- "${CODE_PATHS[@]}") ]]; then
    log "ERROR: $VLAB has uncommitted changes; commit them so the run records the code it uses."
    git -C "$VLAB" status --short -- "${CODE_PATHS[@]}"
    exit 1
  fi
  mkdir -p "$CODE_DIR"
  git -C "$VLAB" archive HEAD src assets accelerate_configs examples | tar -x -C "$CODE_DIR"
  find "$CODE_DIR" -name __pycache__ -type d -prune -exec rm -rf {} +
  git -C "$VLAB" rev-parse HEAD > "$CODE_DIR/COMMIT"
  git -C "$VLAB" rev-parse --abbrev-ref HEAD >> "$CODE_DIR/COMMIT"
  uv pip freeze --python /data/pulkitag/misc/nkim/envs/vlab/bin/python > "$CODE_DIR/pip_freeze.txt" 2>/dev/null \
    || /data/pulkitag/misc/nkim/envs/vlab/bin/python -m pip freeze > "$CODE_DIR/pip_freeze.txt" 2>/dev/null \
    || log "WARNING: could not record the package versions"
  log "snapshotted VLAb $(head -1 "$CODE_DIR/COMMIT") into $CODE_DIR"
fi
log "code: $(tr '\n' ' ' < "$CODE_DIR/COMMIT")(snapshot in $CODE_DIR)"

# ---- Environment ----------------------------------------------------------------------------------------
source /data/pulkitag/misc/nkim/envs/vlab/bin/activate
export PYTHONPATH=$CODE_DIR/src
export HF_HOME=/data/scratch/nkim4724/hf_cache HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton_nkim4724
export TOKENIZERS_PARALLELISM=false
if [[ $WANDB == true ]]; then
  WANDB_API_KEY=$(cat /data/pulkitag/misc/nkim/.secrets/wandb_key)
  export WANDB_API_KEY
fi
ulimit -n "$(ulimit -Hn)"   # hundreds of datasets x workers keep many video files open
cd "$CODE_DIR"

# ---- Fresh start or resume ------------------------------------------------------------------------------
# Newest checkpoint with all files present. Saves are atomic (written to .tmp_* and renamed), so this is
# normally `last`; the file checks guard against anything else.
latest_complete_checkpoint() {
  local d
  for d in "$CKPTS/last" $(ls -d "$CKPTS"/[0-9]*/ 2>/dev/null | sort -r); do
    d=${d%/}
    if [[ -f $d/training_state/training_step.json && -f $d/training_state/optimizer_state.safetensors \
          && -f $d/pretrained_model/model.safetensors && -f $d/pretrained_model/train_config.json ]]; then
      readlink -f "$d"
      return 0
    fi
  done
  return 1
}

if CKPT=$(latest_complete_checkpoint); then
  log "resuming from $CKPT (step $(grep -o '[0-9]\+' "$CKPT/training_state/training_step.json"))"
  TRAIN_ARGS=(
    --resume=true
    --config_path="$CKPT/pretrained_model/train_config.json"
    --steps="$STEPS"
  )
else
  if [[ -e $OUTPUT_DIR ]]; then
    # Killed before its first checkpoint. train.py would switch to a timestamped dir if this one exists,
    # and later restarts would not find that dir's checkpoints, so move the leftovers aside.
    STALE=$OUTPUT_DIR.stale_$(date +%Y%m%d_%H%M%S)
    log "no complete checkpoint in $OUTPUT_DIR; moving it to $STALE"
    mv "$OUTPUT_DIR" "$STALE"
  fi
  case $DATASETS in
    all) REPO_IDS=$(tr -d '\n' < examples/all_datasets_relative.txt) ;;
    smoke) REPO_IDS=community_dataset_v1/Beegbrain/pick_lemon_and_drop_in_bowl,community_dataset_v1/Chojins/chess_game_000_white_red ;;
    *) log "ERROR: DATASETS must be 'all' or 'smoke', got '$DATASETS'"; exit 1 ;;
  esac
  log "fresh start: DATASETS=$DATASETS steps=$STEPS warmup=$WARMUP_STEPS decay=$DECAY_STEPS batch/gpu=$BATCH_SIZE"
  TRAIN_ARGS=(
    --dataset.repo_id="$REPO_IDS"
    --dataset.root=/data/scratch/nkim4724/datasets
    --dataset.video_backend=pyav --dataset.use_imagenet_stats=false
    --dataset.min_fps=30 --dataset.max_fps=30
    --policy.type=smolvla2 --policy.push_to_hub=false --policy.use_amp=false
    --policy.load_vlm_weights=true --policy.train_expert_only=true
    --policy.freeze_vision_encoder=true --policy.state_to_prefix=true
    --policy.num_vlm_layers=16 --policy.expert_width_multiplier=0.75
    --policy.attention_mode=cross_attn --policy.self_attn_every_n_layers=2
    --policy.causal_action_attention_mask=true
    --policy.max_action_dim=32 --policy.max_state_dim=32
    --policy.optimizer_lr=1e-4 --policy.scheduler_warmup_steps="$WARMUP_STEPS"
    --policy.scheduler_decay_steps="$DECAY_STEPS" --policy.scheduler_decay_lr=2.5e-6
    --batch_size="$BATCH_SIZE" --gradient_accumulation_steps=1 --steps="$STEPS"
    --log_freq="$LOG_FREQ" --save_freq="$SAVE_FREQ"
    --keep_last_checkpoints="$KEEP_LAST" --keep_checkpoint_every="$KEEP_EVERY"
    --num_workers="$NUM_WORKERS"
    --wandb.enable="$WANDB" --wandb.project=smolvla-baseline --wandb.disable_artifact=true
    --job_name="$RUN_NAME"
    --output_dir="$OUTPUT_DIR"
  )
fi

# ---- Requeue before the time limit ----------------------------------------------------------------------
requeue_before_time_limit() {
  log "time limit in <10 min: requeueing job $SLURM_JOB_ID (resumes from the last checkpoint)"
  scontrol requeue "$SLURM_JOB_ID"
}
trap requeue_before_time_limit USR1

# ---- Train ----------------------------------------------------------------------------------------------
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
log "launching $NUM_GPUS processes, global batch $((NUM_GPUS * BATCH_SIZE))"
# Port derived from the job ID so two jobs on the same node don't collide on accelerate's default 29500.
accelerate launch --config_file accelerate_configs/multi_gpu.yaml --num_processes="$NUM_GPUS" \
  --main_process_port=$((20000 + SLURM_JOB_ID % 20000)) --mixed_precision bf16 \
  src/lerobot/scripts/train.py "${TRAIN_ARGS[@]}" &
TRAIN_PID=$!
# `wait` returns early when the USR1 trap fires; keep waiting until training exits or Slurm kills the job.
while true; do
  wait "$TRAIN_PID"
  STATUS=$?
  kill -0 "$TRAIN_PID" 2>/dev/null || break
done
log "training exited with status $STATUS"
exit "$STATUS"
