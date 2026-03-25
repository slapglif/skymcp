# Parallelism Strategy Guide

Comprehensive reference for all distributed training parallelism strategies. Includes memory analysis, communication costs, and decision frameworks.

## Parallelism Taxonomy

```
Distributed Training Strategies
|
+-- Data Parallelism (replicate model, shard data)
|   +-- DDP (Distributed Data Parallel)
|   +-- FSDP (Fully Sharded Data Parallel) = ZeRO-3
|   +-- DeepSpeed ZeRO (stages 1/2/3)
|
+-- Model Parallelism (shard model, replicate data)
|   +-- Tensor Parallel (TP) -- split layers horizontally
|   +-- Pipeline Parallel (PP) -- split layers vertically
|   +-- Expert Parallel (EP) -- shard MoE experts
|
+-- Sequence Parallelism
|   +-- Context Parallel (CP) -- split long sequences
|   +-- Ring Attention
|
+-- Hybrid (combine multiple strategies)
    +-- 3D Parallelism: TP + PP + DP
    +-- FSDP + TP (torchtitan pattern)
    +-- DeepSpeed + Megatron (Megatron-DeepSpeed)
```

## Data Parallelism Deep Dive

### DDP (Distributed Data Parallel)

Each GPU holds a complete copy of the model. Data is sharded across GPUs.

**Memory per GPU:**
```
model_params + gradients + optimizer_states
= P + P + (2P for Adam or 4P for mixed-precision Adam)
= 4P to 6P (where P = parameter memory in bytes)
```

**Example: 7B model in BF16**
```
Parameters: 7B * 2 bytes = 14 GB
Gradients:  14 GB
Optimizer:  7B * 4 bytes * 2 (Adam m + v) = 56 GB
Total:      84 GB per GPU
```

**Communication:** AllReduce gradients after each backward pass.
- Volume: 2 * P per step (ring AllReduce)
- Overlap with backward: yes (DDP gradient bucketing)

**When to use:** Model + optimizer fit in single GPU memory.

### FSDP / ZeRO Stages

Progressive memory savings by sharding different components.

| Component | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3/FSDP |
|-----------|-----|--------|--------|-------------|
| Parameters | Full | Full | Full | Sharded (1/N) |
| Gradients | Full | Full | Sharded (1/N) | Sharded (1/N) |
| Optimizer | Full | Sharded (1/N) | Sharded (1/N) | Sharded (1/N) |

**Memory per GPU (N GPUs):**
```
DDP:     P + P + O           (O = optimizer memory = 2-4x P)
ZeRO-1:  P + P + O/N
ZeRO-2:  P + G/N + O/N       (G = gradient memory = P)
ZeRO-3:  P/N + G/N + O/N
```

**Example: 7B model in BF16, 8 GPUs**
```
DDP:     14 + 14 + 56 = 84 GB
ZeRO-1:  14 + 14 + 7  = 35 GB
ZeRO-2:  14 + 1.75 + 7 = 22.75 GB
ZeRO-3:  1.75 + 1.75 + 7 = 10.5 GB
```

**Communication costs:**
```
DDP:     AllReduce gradients (2P)
ZeRO-1:  AllReduce gradients (2P) -- same as DDP
ZeRO-2:  ReduceScatter gradients (P) + AllGather params for backward (P)
ZeRO-3:  ReduceScatter grads (P) + AllGather params for forward AND backward (2P)
```

ZeRO-3 has 1.5x the communication volume of DDP. The tradeoff is memory savings for communication overhead.

## Tensor Parallelism (TP)

Split individual layers across GPUs. Each GPU holds a column or row slice of weight matrices.

**How it works (for a linear layer W*x):**
```
Column Parallel:
  W = [W1 | W2 | ... | WN]  (split columns across N GPUs)
  GPU_i computes: y_i = W_i * x
  AllReduce to combine: y = sum(y_i)

Row Parallel:
  W = [W1; W2; ...; WN]  (split rows across N GPUs)
  GPU_i computes: y_i = W_i * x_i  (x also split)
  AllReduce to combine
```

**For Transformers:**
- Attention: split heads across GPUs (natural parallelism)
- MLP: column-parallel on first linear, row-parallel on second
- Communication: 2 AllReduce per transformer block (forward + backward)

**Memory per GPU:**
```
Parameters: P/TP
Gradients:  P/TP
Optimizer:  O/TP
Activations: reduced by TP (attention heads split)
```

**When to use:**
- Large models that benefit from intra-node NVLink speed
- TP degree = number of GPUs per node (typically 4 or 8)
- DO NOT use TP across nodes (AllReduce latency kills performance)

**Communication:**
- 2 AllReduce per transformer layer per micro-batch
- Requires low-latency interconnect (NVLink, not ethernet)

## Pipeline Parallelism (PP)

Split model layers across GPUs sequentially.

```
GPU 0: Layers 0-7    GPU 1: Layers 8-15
GPU 2: Layers 16-23  GPU 3: Layers 24-31

Data flows: GPU 0 -> GPU 1 -> GPU 2 -> GPU 3
```

**Pipeline bubble problem:**
With naive pipelining, GPUs sit idle waiting for activations.

**Solutions:**
- **Micro-batching (GPipe):** Split batch into micro-batches, pipeline them through.
  - Bubble fraction: (PP - 1) / (PP - 1 + num_microbatches)
  - Example: PP=4, 16 microbatches: bubble = 3/19 = 15.8%
- **Interleaved scheduling (1F1B):** Alternate forward and backward passes.
  - Reduces memory footprint
  - Steady-state: 1 forward + 1 backward per micro-batch
- **Zero-bubble PP:** Advanced scheduling that eliminates pipeline bubbles (2024 research).

**Memory per GPU:**
```
Parameters: P/PP (each GPU holds a subset of layers)
Activations: proportional to num_microbatches (stored for backward)
```

**Communication:**
- Point-to-point send/receive of activations between adjacent stages
- Volume: activation_size * num_microbatches
- Can work over ethernet (latency-tolerant, unlike TP)

## Expert Parallelism (EP)

For Mixture-of-Experts (MoE) models. Shard experts across GPUs.

```
GPU 0: Experts 0-3   GPU 1: Experts 4-7
GPU 2: Experts 8-11  GPU 3: Experts 12-15

Each token is routed to top-k experts.
All-to-All communication shuffles tokens to the right GPU.
```

**Memory per GPU:**
```
Shared parameters: full copy (attention, embeddings)
Expert parameters: E/EP (E = total expert params, EP = expert parallelism degree)
```

**Communication:**
- All-to-All for token routing (forward and backward)
- Volume depends on expert selection pattern

## Context Parallelism (CP)

Split long sequences across GPUs. Each GPU processes a chunk of the sequence.

```
Sequence length: 128K tokens
CP degree: 4
Each GPU processes: 32K tokens

Attention: Ring Attention pattern
- Each GPU computes local attention on its chunk
- KV pairs are passed around the ring
- After full ring rotation, each GPU has seen all KV pairs
```

**When to use:**
- Extremely long sequences (32K+ tokens)
- Model fits on one GPU but sequence doesn't
- Combined with other parallelism (CP + TP + DP)

**Communication:**
- Ring of KV pair transfers
- Volume: proportional to sequence_length / CP

## 3D Parallelism (TP + PP + DP)

For training the largest models (100B+). Combines all three strategies.

```
Example: 256 GPUs (32 nodes x 8 GPUs)

TP = 8  (within each node, using NVLink)
PP = 4  (across 4 nodes per pipeline)
DP = 8  (8 pipeline replicas)

Total: 8 * 4 * 8 = 256 GPUs

Topology:
  Node 0:  [TP group 0, PP stage 0] -- DP replica 0
  Node 1:  [TP group 0, PP stage 1] -- DP replica 0
  Node 2:  [TP group 0, PP stage 2] -- DP replica 0
  Node 3:  [TP group 0, PP stage 3] -- DP replica 0
  Node 4:  [TP group 1, PP stage 0] -- DP replica 1
  ...
```

**Design rules:**
1. TP within nodes (needs NVLink)
2. PP across small node groups (tolerates higher latency)
3. DP across remaining GPUs (AllReduce gradients)
4. Total GPUs = TP * PP * DP

## Framework Comparison

| Feature | PyTorch FSDP | DeepSpeed | Megatron-LM | NeMo |
|---------|-------------|-----------|-------------|------|
| Data Parallel | FSDP (ZeRO-3) | ZeRO 1/2/3 | Distributed DP | All |
| Tensor Parallel | Via DTensor | Limited | Native TP | Native TP |
| Pipeline Parallel | Not native | Limited | Native PP | Native PP |
| Expert Parallel | Not native | MoE support | Native EP | Native EP |
| Context Parallel | Via Ring Attention | Not native | Native CP | Native CP |
| CPU Offload | Yes | ZeRO-Offload | No | No |
| NVMe Offload | No | ZeRO-Infinity | No | No |
| Ease of Use | High | Medium | Low | Medium |
| Max Scale | ~70B comfortable | ~70B comfortable | 1T+ | 1T+ |

**Recommendations:**
- **Under 13B:** FSDP or DeepSpeed ZeRO-2 (either works)
- **13B-70B:** FSDP2 or DeepSpeed ZeRO-3
- **70B-200B:** Megatron-LM or NeMo with TP+PP
- **200B+:** Megatron-LM with full 3D parallelism

## Memory Estimation Formula

For planning GPU requirements:

```python
def estimate_memory_gb(
    params_billions: float,
    dtype_bytes: int = 2,       # 2 for bf16/fp16, 4 for fp32
    optimizer: str = "adam",     # adam uses 2 states per param
    zero_stage: int = 0,
    num_gpus: int = 1,
    sequence_length: int = 2048,
    batch_size: int = 1,
    hidden_dim: int = 4096,
    num_layers: int = 32,
):
    P = params_billions * 1e9

    # Model parameters
    param_mem = P * dtype_bytes / 1e9  # GB

    # Gradients
    grad_mem = P * dtype_bytes / 1e9

    # Optimizer states (Adam: 2 states in fp32)
    opt_mem = P * 4 * 2 / 1e9

    # Apply ZeRO sharding
    if zero_stage >= 1:
        opt_mem /= num_gpus
    if zero_stage >= 2:
        grad_mem /= num_gpus
    if zero_stage >= 3:
        param_mem /= num_gpus

    # Activations (rough estimate)
    # Per-layer: ~12 * hidden * seq_len * batch * dtype_bytes
    act_mem = 12 * hidden_dim * sequence_length * batch_size * dtype_bytes * num_layers / 1e9

    total = param_mem + grad_mem + opt_mem + act_mem
    return {
        "params_gb": round(param_mem, 1),
        "grads_gb": round(grad_mem, 1),
        "optimizer_gb": round(opt_mem, 1),
        "activations_gb": round(act_mem, 1),
        "total_gb": round(total, 1),
    }

# Example: 7B model on 8 GPUs with ZeRO-3
print(estimate_memory_gb(7, zero_stage=3, num_gpus=8))
# {'params_gb': 1.8, 'grads_gb': 1.8, 'optimizer_gb': 7.0, 'activations_gb': 6.3, 'total_gb': 16.8}
```

## Scaling Efficiency

Theoretical vs actual speedup when adding GPUs.

| Strategy | Theoretical Scaling | Typical Efficiency |
|----------|--------------------|--------------------|
| DDP (1 node) | Linear | 90-95% |
| DDP (multi-node) | Linear | 70-90% (network dependent) |
| FSDP/ZeRO-2 | Near-linear | 80-90% |
| FSDP/ZeRO-3 | Sub-linear | 60-80% (extra communication) |
| TP (intra-node) | Near-linear | 85-95% (with NVLink) |
| PP | Sub-linear | 70-85% (pipeline bubble) |
| 3D Parallelism | Complex | 60-80% at scale |

**Key insight:** Communication overhead grows with GPU count. The network becomes the bottleneck at scale. High-bandwidth interconnects (InfiniBand, NVLink, EFA) are essential.

## Profiling Distributed Training

```bash
# NCCL debug logging
export NCCL_DEBUG=INFO

# PyTorch profiler with NCCL tracing
python -c "
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler('./logs/profiler'),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step in range(5):
        train_step()
        prof.step()
"

# View in TensorBoard
tensorboard --logdir ./logs/profiler
```

**What to look for:**
- Communication time vs compute time ratio
- AllReduce blocking on slow nodes
- Pipeline bubble waste
- Memory fragmentation
- GPU utilization < 80% suggests communication bottleneck
