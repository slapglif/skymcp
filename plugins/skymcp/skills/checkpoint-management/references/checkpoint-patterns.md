# Checkpoint Patterns for Distributed Training

Detailed checkpoint strategies for single-GPU, multi-GPU, and multi-node setups. Covers standard PyTorch, FSDP, DeepSpeed, and NeMo.

## Single-GPU Checkpointing

The simplest case. Save model, optimizer, scheduler, and step number.

```python
import os
import torch

def save_checkpoint(step, model, optimizer, scheduler, checkpoint_dir):
    """Atomic checkpoint save."""
    path = os.path.join(checkpoint_dir, f"checkpoint-{step}")
    tmp_path = path + ".tmp"
    os.makedirs(tmp_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(tmp_path, "model.safetensors"))
    torch.save(optimizer.state_dict(), os.path.join(tmp_path, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(tmp_path, "scheduler.pt"))
    torch.save({"step": step}, os.path.join(tmp_path, "training_state.pt"))

    # Atomic rename
    os.rename(tmp_path, path)

    # Update latest pointer
    latest = os.path.join(checkpoint_dir, "latest")
    with open(latest + ".tmp", "w") as f:
        f.write(str(step))
    os.replace(latest + ".tmp", latest)


def load_checkpoint(checkpoint_dir, model, optimizer, scheduler):
    """Load latest checkpoint. Returns step number or 0 if no checkpoint."""
    latest_path = os.path.join(checkpoint_dir, "latest")
    if not os.path.exists(latest_path):
        return 0

    with open(latest_path) as f:
        step = int(f.read().strip())

    path = os.path.join(checkpoint_dir, f"checkpoint-{step}")
    model.load_state_dict(torch.load(os.path.join(path, "model.safetensors")))
    optimizer.load_state_dict(torch.load(os.path.join(path, "optimizer.pt")))
    scheduler.load_state_dict(torch.load(os.path.join(path, "scheduler.pt")))

    return step
```

**Key principles:**
- Atomic writes: save to `.tmp`, then rename. Prevents corruption on crash.
- Latest pointer: simple text file pointing to most recent step number.
- Full state: model + optimizer + scheduler + step. All are needed for exact resumption.

## PyTorch DDP Checkpointing

Only rank 0 saves (all ranks have identical weights in DDP).

```python
import torch.distributed as dist

def save_checkpoint_ddp(step, model, optimizer, scheduler, checkpoint_dir):
    """Save checkpoint from rank 0 only."""
    if dist.get_rank() != 0:
        dist.barrier()  # wait for rank 0 to finish saving
        return

    # Unwrap DDP wrapper
    state_dict = model.module.state_dict()
    save_checkpoint(step, state_dict, optimizer, scheduler, checkpoint_dir)

    dist.barrier()  # signal other ranks that save is complete


def load_checkpoint_ddp(checkpoint_dir, model, optimizer, scheduler):
    """Load checkpoint on all ranks. Rank 0 loads, then broadcasts."""
    step = 0
    if dist.get_rank() == 0:
        step = load_checkpoint(checkpoint_dir, model.module, optimizer, scheduler)

    # Broadcast step number to all ranks
    step_tensor = torch.tensor([step], device="cuda")
    dist.broadcast(step_tensor, src=0)
    step = step_tensor.item()

    if step > 0 and dist.get_rank() != 0:
        # Non-rank-0 loads from saved checkpoint
        path = os.path.join(checkpoint_dir, f"checkpoint-{step}")
        model.module.load_state_dict(torch.load(os.path.join(path, "model.safetensors")))
        optimizer.load_state_dict(torch.load(os.path.join(path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(path, "scheduler.pt")))

    return step
```

## FSDP Checkpointing

FSDP shards model parameters across GPUs. Two save strategies:

### Full State Dict (Consolidate on Rank 0)

Simpler. Produces a single checkpoint that any number of GPUs can load.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

def save_fsdp_full(step, model, optimizer, checkpoint_dir):
    """Consolidate sharded state to rank 0 and save."""
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state = model.state_dict()
        opt_state = FSDP.optim_state_dict(model, optimizer)

    if dist.get_rank() == 0:
        path = os.path.join(checkpoint_dir, f"checkpoint-{step}")
        os.makedirs(path, exist_ok=True)
        torch.save(state, os.path.join(path, "model.pt"))
        torch.save(opt_state, os.path.join(path, "optimizer.pt"))

    dist.barrier()
```

**Pros:** Portable, any GPU count can load.
**Cons:** Slow for very large models (all weights gathered to rank 0).

### Sharded State Dict (Each Rank Saves Its Shard)

Faster for large models. Each rank saves its own shard in parallel.

```python
from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig
from torch.distributed.checkpoint import save, load

def save_fsdp_sharded(step, model, optimizer, checkpoint_dir):
    """Each rank saves its own shard in parallel."""
    cfg = ShardedStateDictConfig(offload_to_cpu=True)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, cfg):
        state = {"model": model.state_dict(), "optimizer": FSDP.optim_state_dict(model, optimizer)}

    path = os.path.join(checkpoint_dir, f"checkpoint-{step}")
    save(state, checkpoint_dir=path)
    dist.barrier()


def load_fsdp_sharded(step, model, optimizer, checkpoint_dir):
    """Each rank loads its own shard."""
    cfg = ShardedStateDictConfig(offload_to_cpu=True)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, cfg):
        state = {"model": model.state_dict(), "optimizer": FSDP.optim_state_dict(model, optimizer)}
        path = os.path.join(checkpoint_dir, f"checkpoint-{step}")
        load(state, checkpoint_dir=path)
        model.load_state_dict(state["model"])
        FSDP.optim_state_dict_to_load(model, optimizer, state["optimizer"])

    dist.barrier()
```

**Pros:** Fast (parallel I/O), no memory bottleneck.
**Cons:** Must load with same or compatible sharding. Less portable.

### FSDP2 (PyTorch 2.5+)

```python
from torch.distributed.checkpoint import save, load
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

def save_fsdp2(step, model, optimizer, checkpoint_dir):
    model_state, opt_state = get_state_dict(model, optimizer)
    path = os.path.join(checkpoint_dir, f"checkpoint-{step}")
    save({"model": model_state, "optimizer": opt_state}, checkpoint_dir=path)

def load_fsdp2(step, model, optimizer, checkpoint_dir):
    model_state, opt_state = get_state_dict(model, optimizer)
    path = os.path.join(checkpoint_dir, f"checkpoint-{step}")
    load({"model": model_state, "optimizer": opt_state}, checkpoint_dir=path)
    set_state_dict(model, optimizer, model_state_dict=model_state, optim_state_dict=opt_state)
```

## DeepSpeed Checkpointing

DeepSpeed has built-in checkpoint management.

### ZeRO Stage 1/2

```python
# Save (automatic sharding across ranks)
model_engine.save_checkpoint(checkpoint_dir, tag=f"step-{step}")

# Load
model_engine.load_checkpoint(checkpoint_dir, tag=f"step-{step}")
```

### ZeRO Stage 3

Stage 3 shards parameters, gradients, AND optimizer across ranks.

```python
# Save -- each rank saves its shard
model_engine.save_checkpoint(checkpoint_dir, tag=f"step-{step}")

# Load -- must have same number of ranks
model_engine.load_checkpoint(checkpoint_dir, tag=f"step-{step}")

# Convert ZeRO-3 checkpoint to single FP32 file (for deployment)
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir, "model.pt", tag=f"step-{step}")
```

**Important:** ZeRO-3 checkpoints require the same number of GPUs to load. To change GPU count, first convert to FP32, then load on new GPU count.

### DeepSpeed Universal Checkpoint

Enables loading across different ZeRO stages and GPU counts.

```python
# Save with universal format
ds_config = {
    "checkpoint": {
        "tag_latest": True,
        "use_node_local_storage": False,
    }
}

# Convert existing checkpoint to universal format
from deepspeed.checkpoint import DeepSpeedCheckpoint
ds_checkpoint = DeepSpeedCheckpoint(checkpoint_dir)
ds_checkpoint.convert_to_universal(output_dir)
```

## NeMo Distributed Checkpointing

NVIDIA NeMo uses a specialized distributed checkpoint format for maximum throughput.

```yaml
# NeMo config (trainer section)
exp_manager:
  checkpoint_callback_params:
    save_top_k: 3
    monitor: val_loss
    mode: min
    every_n_train_steps: 1000
    save_last: true
    async_saving: true    # Non-blocking saves
```

**NeMo features:**
- Fully Parallel Saving (FPS): each GPU rank writes its shard independently
- Async saving: copies to CPU, saves in background while training continues
- Flexible resumption: can resume with different TP/PP configurations
- Automatic latest checkpoint tracking

## Checkpoint Retention Policies

Keeping every checkpoint is expensive. Implement retention policies.

### Rolling Window
Keep last N checkpoints, delete older ones.

```python
import glob
import shutil

def cleanup_checkpoints(checkpoint_dir, keep_last=3):
    """Keep only the N most recent checkpoints."""
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1])
    )
    for old in checkpoints[:-keep_last]:
        shutil.rmtree(old)
```

### Best + Recent
Keep the best checkpoint (by validation loss) plus the last N.

```python
def cleanup_best_recent(checkpoint_dir, val_losses, keep_recent=3):
    """Keep best checkpoint + last N recent."""
    best_step = min(val_losses, key=val_losses.get)
    all_ckpts = sorted(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1])
    )
    recent = set(all_ckpts[-keep_recent:])
    best = os.path.join(checkpoint_dir, f"checkpoint-{best_step}")

    for ckpt in all_ckpts:
        if ckpt not in recent and ckpt != best:
            shutil.rmtree(ckpt)
```

### Exponential Spacing
Keep checkpoints at exponentially increasing intervals.

```python
def should_keep(step, steps_per_save, total_steps):
    """Keep checkpoints at steps: 1, 2, 4, 8, 16, 32, ... plus last 3."""
    keep_steps = set()
    s = steps_per_save
    while s < total_steps:
        keep_steps.add(s)
        s *= 2
    # Always keep last 3
    for i in range(3):
        keep_steps.add(total_steps - i * steps_per_save)
    return step in keep_steps
```

## SkyPilot-Specific Patterns

### Preemption-Safe Checkpointing

```yaml
file_mounts:
  /checkpoints:
    name: training-ckpts-${SKYPILOT_TASK_ID}
    store: s3
    mode: MOUNT

run: |
  # SKYPILOT_TASK_ID is stable across preemptions
  CKPT_DIR=/checkpoints/${SKYPILOT_TASK_ID}

  # Find latest checkpoint for resumption
  LATEST=$(ls -t ${CKPT_DIR}/checkpoint-* 2>/dev/null | head -1)

  python train.py \
    --checkpoint-dir ${CKPT_DIR} \
    ${LATEST:+--resume-from ${LATEST}}
```

### Cross-Job Checkpoint Transfer

Use one job's output as another job's input via shared storage.

```yaml
# Job A: Training
file_mounts:
  /checkpoints:
    name: shared-ckpts
    store: s3
    mode: MOUNT
run: |
  python train.py --checkpoint-dir /checkpoints/training/

---
# Job B: Evaluation (separate YAML)
file_mounts:
  /checkpoints:
    name: shared-ckpts
    store: s3
    mode: MOUNT_CACHED    # Read-only cached access
run: |
  LATEST=$(ls -t /checkpoints/training/checkpoint-* | head -1)
  lm_eval --model hf --model_args pretrained=${LATEST} --tasks mmlu
```

## Verification Checklist

Before relying on a checkpointing strategy:

- [ ] Checkpoint includes: model weights, optimizer state, scheduler state, step number, RNG state
- [ ] Writes are atomic (temp dir + rename)
- [ ] Latest pointer is updated after successful write
- [ ] Resume produces identical training trajectory (deterministic)
- [ ] Checkpoint works after spot preemption (test manually)
- [ ] Retention policy prevents unbounded storage growth
- [ ] Checkpoint can be loaded on different GPU counts (if using FSDP/DeepSpeed)
