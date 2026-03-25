# Training Diagnosis Playbook

Step-by-step procedures for diagnosing and resolving common training failures.

## Procedure 1: Out of Memory (OOM)

### Symptoms

```
RuntimeError: CUDA out of memory. Tried to allocate X GiB (GPU Y; Z GiB total capacity; W GiB already allocated)
```

Or: Process killed by OS (OOM killer), no CUDA error message.

### Step 1: Identify Memory Usage

```python
import torch

def print_memory_stats():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f"GPU {i}: {allocated:.2f} GB allocated, "
              f"{reserved:.2f} GB reserved, {total:.2f} GB total")

# Call before and after model creation, before and after first forward pass
print_memory_stats()
```

### Step 2: Identify the Phase

OOM can occur during:
1. **Model loading** -- Model too large for GPU
2. **Forward pass** -- Activations too large (batch size, sequence length)
3. **Backward pass** -- Gradients + activations exceed memory
4. **Optimizer step** -- Optimizer states (AdamW uses 2x model size for moments)

### Step 3: Resolution Cascade

Try each fix in order. Stop when OOM is resolved.

**Fix 1: Reduce batch size**
```python
# Halve micro_batch_size, double gradient_accumulation_steps
# Effective batch size stays the same
micro_batch_size = 1  # was 2
gradient_accumulation_steps = 8  # was 4
```

**Fix 2: Enable gradient checkpointing**
```python
# Trades ~30% compute for ~60% activation memory savings
model.gradient_checkpointing_enable()

# For HuggingFace models
from transformers import TrainingArguments
args = TrainingArguments(gradient_checkpointing=True)
```

**Fix 3: Switch to mixed precision**
```python
# BF16 halves activation and weight memory
from torch.amp import autocast, GradScaler
scaler = GradScaler()

with autocast("cuda", dtype=torch.bfloat16):
    output = model(input_ids)
    loss = criterion(output, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Fix 4: Use LoRA/QLoRA**
```python
# Only adapter parameters need optimizer states
# Reduces memory from ~14 bytes/param to ~14 bytes/LoRA_param
from peft import get_peft_model, LoraConfig

config = LoraConfig(r=32, lora_alpha=64, target_modules="all-linear")
model = get_peft_model(model, config)
print(f"Trainable: {model.print_trainable_parameters()}")
```

**Fix 5: DeepSpeed ZeRO-3 + CPU offload**
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu", "pin_memory": true},
    "offload_param": {"device": "cpu", "pin_memory": true}
  }
}
```

**Fix 6: Upgrade GPU**
```bash
# SkyPilot makes this trivial
sky launch train.yaml --gpus A100-80GB:4  # was A100-40GB:4
# Or
sky launch train.yaml --gpus H100:4
```

### Step 4: Verify Fix

```python
# Monitor peak memory during training
torch.cuda.reset_peak_memory_stats()

# Run one full training step
loss = train_step(batch)
loss.backward()
optimizer.step()

peak = torch.cuda.max_memory_allocated() / 1e9
total = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f"Peak: {peak:.2f} GB / {total:.2f} GB ({peak/total*100:.1f}%)")
# Target: 80-90% utilization (not 95%+)
```

## Procedure 2: NaN Loss

### Symptoms

```
Step 1000: loss = nan
Step 1001: loss = nan
# Model outputs become garbage
```

### Step 1: Find When NaN Started

```python
# Add NaN detection to training loop
for step, batch in enumerate(dataloader):
    loss = train_step(batch)

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"NaN/Inf detected at step {step}")
        print(f"  Loss: {loss.item()}")
        print(f"  LR: {scheduler.get_last_lr()}")

        # Check model parameters
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"  NaN grad in: {name}")
                grad_norm = param.grad.norm().item()
                if grad_norm > 100:
                    print(f"  Exploding grad in: {name} (norm={grad_norm:.2f})")

        # Check input data
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                if torch.isnan(val).any():
                    print(f"  NaN in input: {key}")
                if torch.isinf(val).any():
                    print(f"  Inf in input: {key}")
        break
```

### Step 2: Common Causes and Fixes

**Cause 1: Learning rate too high**
```python
# Reduce LR by 10x
optimizer = AdamW(model.parameters(), lr=1e-5)  # was 1e-4

# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Increase warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=500,  # was 100
    num_training_steps=total_steps,
)
```

**Cause 2: FP16 loss scaling overflow**
```python
# Switch from FP16 to BF16 (no loss scaling needed)
# BF16 has same exponent range as FP32, just lower mantissa precision
model = model.to(torch.bfloat16)
```

**Cause 3: Corrupted data**
```python
# Validate data before training
for i, batch in enumerate(dataloader):
    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            if torch.isnan(val).any() or torch.isinf(val).any():
                print(f"Bad data in batch {i}, field {key}")
```

**Cause 4: Numerical instability in softmax/attention**
```python
# Use FlashAttention (numerically stable softmax in hardware)
# pip install flash-attn
from flash_attn import flash_attn_func

# Or use PyTorch's scaled_dot_product_attention
output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**Cause 5: Division by zero in loss**
```python
# Add epsilon to denominators
loss = -torch.log(probs + 1e-8)

# Check for empty batches
if labels.numel() == 0:
    continue
```

### Step 3: Recovery

```python
# Load last known good checkpoint
checkpoint = torch.load("/checkpoints/step_900")  # Last good step
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])

# Reduce LR by 5-10x from the checkpoint's LR
for pg in optimizer.param_groups:
    pg["lr"] *= 0.1

# Resume training
```

## Procedure 3: Loss Plateau

### Symptoms

Loss stops decreasing for 1000+ steps despite training continuing. No NaN, no OOM, just stagnation.

### Step 1: Classify the Plateau

```python
# Is train loss flat? Is val loss flat?
# Check the last 1000 steps

recent_train_losses = train_losses[-1000:]
recent_val_losses = val_losses[-1000:]

train_trend = recent_train_losses[-1] - recent_train_losses[0]
val_trend = recent_val_losses[-1] - recent_val_losses[0]

if train_trend > 0 and val_trend > 0:
    print("Both increasing -> Possible instability")
elif train_trend < -0.01 and val_trend > 0:
    print("Train down, val up -> Overfitting")
elif abs(train_trend) < 0.001 and abs(val_trend) < 0.001:
    print("Both flat -> Converged or stuck")
elif train_trend < -0.001 and val_trend < -0.001:
    print("Both decreasing -> Still learning (be patient)")
```

### Step 2: Check Learning Rate Schedule

```python
import matplotlib.pyplot as plt

# Plot LR schedule
lrs = [scheduler.get_last_lr()[0] for _ in range(total_steps)]
plt.plot(lrs)
plt.axvline(x=current_step, color='r', label='Current step')
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('LR Schedule')
plt.savefig('/tmp/lr_schedule.png')
```

If LR has decayed to near zero, the model cannot learn anymore:
```python
# Option 1: Cosine annealing with warm restarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=2)

# Option 2: Increase minimum LR
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
```

### Step 3: Check Data Quality

```python
# Are we seeing the same data repeatedly?
seen_hashes = set()
duplicates = 0
for batch in dataloader:
    h = hash(batch["input_ids"].tobytes())
    if h in seen_hashes:
        duplicates += 1
    seen_hashes.add(h)

print(f"Duplicate batches: {duplicates}/{len(dataloader)}")
# If > 10%, data pipeline has a bug or dataset is too small
```

### Step 4: Targeted Fixes

**Fix: Weight decay too high**
```python
# Reduce from 0.1 to 0.01
for pg in optimizer.param_groups:
    pg["weight_decay"] = 0.01
```

**Fix: Try a different optimizer**
```python
# Switch from AdamW to Lion (often breaks through plateaus)
from lion_pytorch import Lion
optimizer = Lion(model.parameters(), lr=1e-5, weight_decay=1e-2)
```

**Fix: Increase model capacity**
```python
# If model is too small for the task
# Increase hidden size, add layers, or use a larger base model
```

## Procedure 4: Gradient Explosion

### Symptoms

Gradient norm spikes to 100+. Loss jumps erratically. Training may recover or may diverge.

### Step 1: Monitor Gradient Norms

```python
def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

# In training loop
grad_norm = compute_grad_norm(model)
if grad_norm > 10.0:
    print(f"WARNING: grad_norm={grad_norm:.2f} at step {step}")

    # Find which layers have the largest gradients
    for name, p in model.named_parameters():
        if p.grad is not None:
            layer_norm = p.grad.data.norm(2).item()
            if layer_norm > 1.0:
                print(f"  {name}: {layer_norm:.4f}")
```

### Step 2: Fixes

```python
# Fix 1: Add gradient clipping (always do this)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Fix 2: Lower learning rate
for pg in optimizer.param_groups:
    pg["lr"] *= 0.5

# Fix 3: Increase warmup
# Longer warmup = more gradual LR increase = fewer spikes

# Fix 4: Check for outlier data
# A single batch with extreme values can cause a spike
# Add input normalization or clamp input values
```

## Procedure 5: Slow Throughput

### Step 1: Profile

```bash
# Check GPU utilization
nvidia-smi dmon -s u -d 5
# Look for:
#   SM% (compute utilization) -- should be > 80%
#   Mem% (memory bandwidth) -- should be > 50%
```

```python
# PyTorch profiler
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler("/tmp/profiler"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 5:
            break
        loss = train_step(batch)
        loss.backward()
        optimizer.step()
        prof.step()

# View in TensorBoard
# tensorboard --logdir=/tmp/profiler
```

### Step 2: Common Bottlenecks

**Bottleneck: Data loading (GPU util < 50%, CPU util high)**
```python
# Increase workers
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,          # Increase from default 0
    pin_memory=True,        # Faster CPU->GPU transfer
    prefetch_factor=4,      # Prefetch 4 batches per worker
    persistent_workers=True, # Don't restart workers each epoch
)
```

**Bottleneck: No torch.compile**
```python
# 1.5-3x speedup on H100, 1.2-2x on A100
model = torch.compile(model, mode="reduce-overhead")
```

**Bottleneck: No FlashAttention**
```python
# Verify FlashAttention is active
# Should see "Using flash_attn" in model logs
# If not:
pip install flash-attn --no-build-isolation
```

**Bottleneck: Small batch size**
```python
# GPU kernels are most efficient at large batch sizes
# Increase batch size until GPU memory is 80-90% utilized
# Use gradient accumulation to maintain effective batch size
```

**Bottleneck: Disk I/O (checkpoint saves)**
```yaml
# SkyPilot: use high-tier disk
resources:
  disk_tier: high  # or ultra
```

```python
# Save checkpoints asynchronously
import threading

def async_save(model, path):
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    thread = threading.Thread(target=torch.save, args=(state, path))
    thread.start()
    return thread
```

**Bottleneck: NCCL communication (multi-node)**
```bash
# Debug NCCL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Use high-bandwidth network
# SkyPilot:
resources:
  network_tier: premium  # GCP only
```

### Step 3: Throughput Targets

Use these as sanity checks. If your throughput is significantly below these numbers, there is a bottleneck to find.

| GPU | 7B (bf16) | 13B (bf16) | 70B (bf16, 8-GPU) |
|-----|-----------|------------|-------------------|
| RTX 3060 12GB | ~400 tok/s | OOM | OOM |
| RTX 4090 24GB | ~2,800 tok/s | ~1,400 tok/s | OOM |
| A100 40GB | ~3,000 tok/s | ~1,800 tok/s | ~800 tok/s |
| A100 80GB | ~3,500 tok/s | ~2,200 tok/s | ~1,200 tok/s |
| H100 80GB | ~8,000 tok/s | ~5,000 tok/s | ~2,500 tok/s |

These assume: bf16, FlashAttention, torch.compile, optimal batch size. Full fine-tuning. LoRA is 2-3x faster.
