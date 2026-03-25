---
name: distributed-training
description: Use when setting up multi-GPU or multi-node training, configuring torchrun or deepspeed launchers, choosing parallelism strategy (FSDP, DeepSpeed ZeRO, tensor parallel, pipeline parallel), tuning NCCL, or enabling high-performance networking with SkyPilot
---

# Distributed Training Patterns

## Overview

Scale training beyond a single GPU using data parallelism, model parallelism, or a combination. SkyPilot automates multi-node cluster provisioning, environment variable injection, and high-performance networking setup.

**Core principle:** Start simple (single GPU with gradient accumulation), then scale up only when needed. Each parallelism strategy adds complexity; use the minimum that fits your model and data.

## When to Use

- Model does not fit in a single GPU's memory
- Training is too slow on a single GPU and you need to parallelize
- Need to use multiple nodes (multi-machine training)
- Configuring DeepSpeed ZeRO stages for memory optimization
- Setting up FSDP for PyTorch-native sharding

**Do not use for:**
- Models that fit comfortably on one GPU (use gradient accumulation instead)
- Inference-only workloads (use vLLM tensor parallelism instead)
- Small fine-tuning jobs (QLoRA on single GPU is simpler)

## Decision Guide: Which Parallelism?

| Scenario | Strategy | Why |
|----------|----------|-----|
| Model fits on 1 GPU, need speed | Data Parallel (DDP) | Simplest, near-linear scaling |
| Model fits on 1 GPU with optimizer sharding | DeepSpeed ZeRO-1/2 or FSDP | Reduce optimizer memory |
| Model does NOT fit on 1 GPU | DeepSpeed ZeRO-3 or FSDP (full shard) | Shard parameters across GPUs |
| 70B+ model, single node | ZeRO-3 + CPU offload | Trade compute for memory |
| 70B+ model, multi-node | FSDP2 + network_tier:best | Best PyTorch-native scaling |
| 100B+ model, Megatron | TP + PP + DP | Maximum control for massive models |
| MoE model | Expert Parallel (EP) | Shard experts across GPUs |
| Very long sequences | Context Parallel (CP) | Split sequence across GPUs |

**Flowchart:**
```
Does model fit on 1 GPU?
  |
  +-- YES: Use DDP or gradient accumulation
  |
  +-- NO: Does model fit with ZeRO-2/FSDP (shard optimizer + gradients)?
       |
       +-- YES: Use ZeRO-2 or FSDP
       |
       +-- NO: Use ZeRO-3 or FSDP full shard
            |
            +-- Still OOM? Add CPU offload or use TP+PP
```

## SkyPilot Multi-Node Setup

SkyPilot provides environment variables for distributed training automatically. No manual hostfile or IP discovery needed.

```yaml
num_nodes: 4
resources:
  accelerators: H100:8
  network_tier: best    # Enables InfiniBand/EFA

run: |
  MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)

  torchrun \
    --nnodes=$SKYPILOT_NUM_NODES \
    --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
    --node_rank=$SKYPILOT_NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    train.py
```

**SkyPilot environment variables:**

| Variable | Value | Description |
|----------|-------|-------------|
| `SKYPILOT_NODE_IPS` | Newline-separated IPs | All node IPs (head node first) |
| `SKYPILOT_NUM_NODES` | Integer | Total number of nodes |
| `SKYPILOT_NODE_RANK` | 0, 1, 2, ... | This node's rank (0 = head) |
| `SKYPILOT_NUM_GPUS_PER_NODE` | Integer | GPUs on this node |
| `SKYPILOT_TASK_ID` | String | Stable across preemptions |

## torchrun Multi-Node

The standard PyTorch distributed launcher.

```yaml
num_nodes: 2
resources:
  accelerators: H100:8
  network_tier: best

run: |
  MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)
  NUM_GPUS=$(( $SKYPILOT_NUM_NODES * $SKYPILOT_NUM_GPUS_PER_NODE ))

  torchrun \
    --nnodes=$SKYPILOT_NUM_NODES \
    --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
    --node_rank=$SKYPILOT_NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    train.py \
      --total_gpus $NUM_GPUS
```

**In your training script:**
```python
import torch.distributed as dist
import os

def setup_distributed():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

local_rank = setup_distributed()
```

## DeepSpeed Multi-Node

DeepSpeed has its own launcher that handles multi-node orchestration.

```yaml
num_nodes: 4
resources:
  accelerators: H100:8
  network_tier: best

run: |
  # Build hostfile from SkyPilot env vars
  echo "$SKYPILOT_NODE_IPS" | awk -v gpus=$SKYPILOT_NUM_GPUS_PER_NODE \
    '{print $1 " slots=" gpus}' > /tmp/hostfile

  # Only head node launches deepspeed
  if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
    deepspeed \
      --hostfile /tmp/hostfile \
      --master_addr $(echo "$SKYPILOT_NODE_IPS" | head -n1) \
      --master_port 29500 \
      train.py \
        --deepspeed ds_config.json
  fi
```

**Head-only pattern:** Only node rank 0 runs the launcher. Other nodes wait for the launcher to SSH in and start workers. This is how DeepSpeed and some other launchers work.

```bash
if [ "${SKYPILOT_NODE_RANK}" == "0" ]; then
  # Head-only commands: launch distributed training, start tensorboard, run eval
  deepspeed --hostfile /tmp/hostfile train.py --deepspeed ds_config.json
  tensorboard --logdir /logs --port 6006 &
fi
```

## FSDP Configuration

PyTorch-native Fully Sharded Data Parallel. Preferred over DeepSpeed for new projects using PyTorch 2.x.

### FSDP1 (PyTorch 2.0-2.4)

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

# Define wrapping policy (wrap each transformer layer)
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},  # your model's block class
)

# Mixed precision
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# Wrap model
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mp_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
    device_id=local_rank,
    limit_all_gathers=True,  # reduces memory peak
)
```

### FSDP2 (PyTorch 2.5+)

Cleaner API, better composability with other features.

```python
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)

# Shard each transformer block
for layer in model.layers:
    fully_shard(layer, mp_policy=mp_policy)

# Shard the entire model (root)
fully_shard(model, mp_policy=mp_policy)
```

**FSDP sharding strategies:**

| Strategy | Memory Savings | Communication | Equivalent |
|----------|---------------|---------------|------------|
| NO_SHARD | None | Lowest | DDP |
| SHARD_GRAD_OP | ~2x | Medium | ZeRO-2 |
| FULL_SHARD | ~4x | Highest | ZeRO-3 |
| HYBRID_SHARD | ~2-4x | Medium | ZeRO-3 within node, DDP between nodes |

## DeepSpeed ZeRO Stages

### ZeRO-1 (Shard Optimizer States)

```json
{
  "zero_optimization": {
    "stage": 1
  },
  "bf16": {"enabled": true},
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

**Memory savings:** ~4x optimizer memory reduction. No communication overhead beyond DDP.

### ZeRO-2 (Shard Optimizer + Gradients)

```json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  },
  "bf16": {"enabled": true},
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

**Memory savings:** ~8x vs DDP. Good balance of memory savings and speed.

### ZeRO-3 (Shard Everything)

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "none"},
    "offload_param": {"device": "none"},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "bf16": {"enabled": true},
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

### ZeRO-3 with CPU Offload (Maximum Memory Savings)

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "bf16": {"enabled": true},
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

**Warning:** CPU offload is significantly slower (2-5x). Use only when GPU memory is truly insufficient.

For complete DeepSpeed ZeRO configuration reference with all tunable parameters, see `references/deepspeed-zero-config.md`.

## High-Performance Networking

Multi-node training is bottlenecked by inter-node communication. Use SkyPilot's `network_tier: best` to enable high-bandwidth interconnects.

```yaml
resources:
  accelerators: H100:8
  network_tier: best    # CRITICAL for multi-node
```

**What `network_tier: best` enables:**
- **AWS:** Elastic Fabric Adapter (EFA) -- up to 3200 Gbps
- **GCP:** GPUDirect-TCPX -- up to 200 Gbps per GPU
- **Azure:** InfiniBand NDR -- up to 3200 Gbps

**Without `network_tier: best`:** Standard TCP networking (~25-100 Gbps), which becomes a severe bottleneck for multi-node training.

### NCCL Tuning

NCCL (NVIDIA Collective Communication Library) handles GPU-to-GPU communication.

```bash
# Common NCCL environment variables
export NCCL_DEBUG=INFO               # Debug logging
export NCCL_IB_DISABLE=0            # Enable InfiniBand (default)
export NCCL_NET_GDR_LEVEL=5         # GPUDirect RDMA level
export NCCL_SOCKET_IFNAME=eth0      # Network interface
export NCCL_P2P_DISABLE=0           # Enable P2P (NVLink)
export NCCL_CROSS_NIC=1             # Enable cross-NIC communication
```

**SkyPilot typically sets these automatically** when `network_tier: best` is configured. Only override for debugging.

## Tensor Parallel + Pipeline Parallel

For very large models (100B+) where even FSDP/ZeRO-3 is insufficient.

**Tensor Parallel (TP):** Split individual layers across GPUs.
- Each GPU holds a slice of every layer
- Requires fast inter-GPU communication (NVLink within node)
- Typical: TP=8 (one node)

**Pipeline Parallel (PP):** Split layers sequentially across GPUs.
- GPU 0 holds layers 0-7, GPU 1 holds layers 8-15, etc.
- Micro-batching hides pipeline bubble
- Can span across nodes

**Combined TP + PP + DP:**
```
Example: 128 GPUs (16 nodes x 8 GPUs)
TP=8  (within each node)
PP=4  (across 4 nodes)
DP=4  (4 pipeline replicas)

Total: 8 * 4 * 4 = 128 GPUs
```

This level of parallelism requires frameworks like **Megatron-LM** or **NeMo**. For most models under 70B, FSDP or DeepSpeed ZeRO is sufficient.

## Gradient Accumulation (Single GPU Alternative)

Before scaling to multiple GPUs, consider gradient accumulation.

```python
accumulation_steps = 8
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Effective batch size = micro_batch * accumulation_steps * num_gpus**

This simulates a larger batch size on a single GPU. No communication overhead, no distributed complexity.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Forgetting `network_tier: best` for multi-node | Without it, inter-node communication bottlenecks training. |
| Using DeepSpeed launcher on all nodes | Only head node (rank 0) runs the launcher. Others wait. |
| Not matching batch size to GPU count | Effective batch = micro_batch * accum_steps * world_size. Adjust accordingly. |
| ZeRO-3 + CPU offload by default | Start with ZeRO-2. Only use ZeRO-3 if model doesn't fit. CPU offload is last resort. |
| Using FSDP NO_SHARD (DDP) for large models | NO_SHARD doesn't save memory. Use FULL_SHARD for memory savings. |
| Not testing single-node before multi-node | Debug on 1 node first. Multi-node adds network failure modes. |

## Quick Reference

```bash
# SkyPilot multi-node launch
# In YAML: num_nodes: 4, resources.network_tier: best

# torchrun (in run: section)
MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)
torchrun --nnodes=$SKYPILOT_NUM_NODES \
  --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
  --node_rank=$SKYPILOT_NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=29500 \
  train.py

# DeepSpeed (head node only)
if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
  echo "$SKYPILOT_NODE_IPS" | awk -v g=$SKYPILOT_NUM_GPUS_PER_NODE \
    '{print $1 " slots=" g}' > /tmp/hostfile
  deepspeed --hostfile /tmp/hostfile train.py --deepspeed ds_config.json
fi

# NCCL debug
export NCCL_DEBUG=INFO
```

For detailed parallelism strategies, see `references/parallelism-guide.md`. For complete DeepSpeed ZeRO configuration, see `references/deepspeed-zero-config.md`.
