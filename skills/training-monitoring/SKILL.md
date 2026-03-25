---
name: training-monitoring
description: Use when monitoring training runs, interpreting loss curves, configuring W&B or TensorBoard, diagnosing training failures like NaN loss or OOM or loss plateaus or gradient explosion, tracking training metrics, debugging slow throughput, or deciding when to stop training - the training monitoring and diagnosis reference
---

# Training Monitoring and Diagnosis

## Metrics to Track

Every training run must log these core metrics:

| Metric | Why | Alert Threshold |
|--------|-----|-----------------|
| `train/loss` | Primary training signal | Stops decreasing for > 500 steps |
| `val/loss` | Generalization check | Diverges from train_loss |
| `train/learning_rate` | Verify schedule is correct | Unexpected jumps |
| `train/gradient_norm` | Detect explosion/vanishing | > 10.0 or < 1e-7 |
| `train/throughput_tokens_per_sec` | Training speed | Drops > 20% from baseline |
| `system/gpu_memory_allocated_gb` | Memory pressure | > 95% of total VRAM |
| `system/gpu_utilization_pct` | Compute efficiency | < 60% sustained |
| `train/epoch` | Progress tracking | -- |
| `train/global_step` | Step counter | -- |

## W&B Integration

### Standard Pattern

```python
import wandb
import os

wandb.init(
    project="my-training",
    name=os.environ.get("SKYPILOT_TASK_ID", "local-run"),
    config={
        "model": "llama-3-8b",
        "method": "lora-r32",
        "lr": 2e-4,
        "batch_size": 32,
        "gpus": os.environ.get("SKYPILOT_NUM_GPUS_PER_NODE", 1),
    },
    tags=["finetune", "llama3", "lora"],
)

# In training loop
for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    grad_norm = get_gradient_norm(model)

    wandb.log({
        "train/loss": loss.item(),
        "train/learning_rate": scheduler.get_last_lr()[0],
        "train/gradient_norm": grad_norm,
        "train/throughput_tokens_per_sec": tokens_this_step / step_time,
        "train/global_step": step,
        "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "system/gpu_utilization_pct": get_gpu_util(),
    }, step=step)

    if step % eval_interval == 0:
        val_loss = evaluate(model, val_loader)
        wandb.log({"val/loss": val_loss}, step=step)

wandb.finish()
```

### SkyPilot YAML with W&B

```yaml
envs:
  WANDB_API_KEY: null  # Read from local env
  WANDB_PROJECT: my-training

run: |
  python train.py --wandb-project $WANDB_PROJECT
```

### W&B Alerts

```python
# Alert on training anomalies
if loss > 100.0 or math.isnan(loss):
    wandb.alert(
        title="Training anomaly detected",
        text=f"Loss={loss} at step {step}. Check run immediately.",
        level=wandb.AlertLevel.ERROR,
    )
```

## TensorBoard via SSH Tunnel

For clusters without public TensorBoard:

```bash
# From local machine, tunnel to SkyPilot cluster
ssh -L 6006:localhost:6006 mycluster

# On the cluster
tensorboard --logdir=/checkpoints/logs --port=6006 --bind_all
```

Then open `http://localhost:6006` in your browser.

### TensorBoard with PyTorch

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="/checkpoints/logs")
writer.add_scalar("train/loss", loss.item(), step)
writer.add_scalar("train/lr", lr, step)
writer.add_histogram("gradients/layer0", model.layers[0].weight.grad, step)
writer.close()
```

## NeMo Logging

NeMo provides built-in timing and performance logging:

```python
# In NeMo training config
trainer = NeMoTrainer(
    log_every_n_steps=10,
    callbacks=[
        TimingCallback(),          # Per-step timing
        MemoryProfileCallback(),   # GPU memory tracking
    ],
)
```

Set `timing_log_level`:
- **0**: No timing (fastest)
- **1**: Per-step timing (recommended)
- **2**: Per-microbatch timing (debug only, adds overhead)

TransformerEngine tensor debug: set `TE_DEBUG=1` to log FP8 scaling factors and amax history.

## Diagnosis Flowchart

### Out of Memory (OOM)

Symptoms: `CUDA out of memory`, `RuntimeError: CUDA error`, process killed by OS.

Resolution sequence (try each in order):

1. **Reduce micro batch size** -- Halve it. Compensate with gradient accumulation.
2. **Enable gradient checkpointing** -- Trades 30% compute for 60% memory savings.
   ```python
   model.gradient_checkpointing_enable()
   ```
3. **Switch to mixed precision** -- bf16 halves activation memory.
   ```python
   from torch.amp import autocast
   with autocast("cuda", dtype=torch.bfloat16):
       output = model(input)
   ```
4. **Enable CPU offloading** -- DeepSpeed ZeRO-3 + CPU offload.
   ```json
   {
     "zero_optimization": {
       "stage": 3,
       "offload_optimizer": {"device": "cpu"},
       "offload_param": {"device": "cpu"}
     }
   }
   ```
5. **Use LoRA/QLoRA** -- Only adapter parameters need optimizer states.
6. **Upgrade to larger GPU** -- SkyPilot makes this a one-line change:
   ```bash
   sky launch train.yaml --gpus A100:8  # or H100:8
   ```

Memory debugging tools:
```python
# Print memory snapshot
torch.cuda.memory_summary(device=None, abbreviated=True)

# Find memory leaks
torch.cuda.reset_peak_memory_stats()
# ... run suspect code ...
print(f"Peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### NaN Loss

Symptoms: loss becomes `nan` or `inf`, model outputs garbage.

Diagnosis checklist:

1. **Learning rate too high** -- Reduce by 10x. Check warmup steps.
   ```python
   # Add gradient clipping as safety net
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```
2. **Corrupted data** -- Check for NaN/inf in input tensors.
   ```python
   for name, param in model.named_parameters():
       if torch.isnan(param).any():
           print(f"NaN in {name}")
       if torch.isinf(param).any():
           print(f"Inf in {name}")
   ```
3. **Loss scaling overflow (FP16)** -- Switch to bf16 (no loss scaling needed) or adjust scaler.
   ```python
   # bf16 is almost always preferable to fp16 on modern GPUs
   torch.set_default_dtype(torch.bfloat16)
   ```
4. **Division by zero in loss** -- Add epsilon to denominators, check for empty batches.
5. **Model initialization** -- Verify weight initialization is appropriate for model depth.
6. **Numerical instability in attention** -- Use FlashAttention (numerically stable softmax).

Emergency recovery: load last good checkpoint, reduce LR by 5-10x, resume.

### Loss Plateau

Symptoms: loss stops decreasing for thousands of steps despite training continuing.

Diagnosis checklist:

1. **LR schedule exhausted** -- Check if cosine annealing hit minimum. Try warmup restart.
   ```python
   # Cosine annealing with warm restarts
   scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
       optimizer, T_0=1000, T_mult=2, eta_min=1e-6
   )
   ```
2. **Data exhaustion** -- If dataset is small, model may have memorized it. Add more data or stronger augmentation.
3. **Model capacity** -- Model may be too small for the task. Try increasing width or depth.
4. **Batch size** -- Try reducing batch size (better gradient noise for escaping plateaus).
5. **Weight decay** -- Too high can prevent convergence. Try reducing from 0.1 to 0.01.
6. **Optimizer** -- Switch from SGD to AdamW, or try Lion/Sophia for transformer pretraining.

### Gradient Explosion

Symptoms: gradient norm spikes to 100+, loss jumps, training becomes unstable.

Resolution:

1. **Add gradient clipping** (first line of defense):
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```
2. **Lower learning rate** -- Halve it. Especially important after warmup.
3. **Increase warmup steps** -- Gradual LR increase stabilizes early training.
4. **Check data** -- Outlier batches with extreme values can cause spikes.
5. **Layer normalization** -- Add LayerNorm/RMSNorm if missing. Pre-norm (before attention/FFN) is more stable than post-norm.

### Slow Throughput

Symptoms: tokens/sec well below expected for hardware.

Diagnosis:

1. **Check GPU utilization**:
   ```bash
   watch -n 1 nvidia-smi  # Should show >90% utilization
   ```
2. **Data loading bottleneck** -- If GPU util is low, data loading is the bottleneck.
   ```python
   # Increase dataloader workers
   DataLoader(dataset, num_workers=8, pin_memory=True, prefetch_factor=4)
   ```
3. **torch.compile** -- Enable if not already (2-3x speedup on modern GPUs):
   ```python
   model = torch.compile(model)
   ```
4. **FlashAttention** -- Verify it is active (check for `flash_attn` in logs).
5. **Communication overhead** -- For multi-node, check network bandwidth. Use NCCL_DEBUG=INFO.
6. **Disk I/O** -- Use SkyPilot `disk_tier: high` or `ultra` for checkpoint-heavy workloads.
7. **Batch size too small** -- GPU kernels are inefficient at small batch sizes. Increase until memory is ~80-90% utilized.

## Checkpoint Monitoring

### Checkpoint Strategy

```python
# Save every N steps + keep best K checkpoints
if step % save_interval == 0:
    save_checkpoint(model, optimizer, step, f"/checkpoints/step_{step}")
    # Upload to cloud storage (survives spot preemption)
    os.system(f"aws s3 sync /checkpoints/ s3://my-bucket/checkpoints/")

# Also save on val_loss improvement
if val_loss < best_val_loss:
    best_val_loss = val_loss
    save_checkpoint(model, optimizer, step, "/checkpoints/best")
```

### SkyPilot with MOUNT_CACHED for Checkpoints

```yaml
file_mounts:
  /checkpoints:
    source: s3://my-bucket/checkpoints
    mode: MOUNT_CACHED  # Local cache + async upload

resources:
  use_spot: true
  job_recovery:
    strategy: FAILOVER  # Auto-recover from preemption
```

The training script must handle resume from checkpoint:

```python
# Resume from latest checkpoint
ckpt_path = find_latest_checkpoint("/checkpoints/")
if ckpt_path:
    model, optimizer, start_step = load_checkpoint(ckpt_path)
    print(f"Resuming from step {start_step}")
```

## When to Stop Training

| Signal | Action |
|--------|--------|
| Val loss increasing for > 3 eval cycles | Stop (overfitting) |
| Val loss flat for > 10 eval cycles | Stop (converged) or change LR |
| Train loss flat, val loss flat | Stop (converged) |
| Train loss decreasing, val loss increasing | Stop (overfitting) |
| Train loss still decreasing, val loss still decreasing | Continue |
| Compute budget exhausted | Stop, evaluate, decide |

## GPU Utilization Reference

| GPU | VRAM | bf16 TFLOPS | Expected tok/sec (7B) |
|-----|------|------------|----------------------|
| A100 40GB | 40 GB | 312 | ~3,000 |
| A100 80GB | 80 GB | 312 | ~3,500 |
| H100 80GB | 80 GB | 990 | ~8,000 |
| H200 141GB | 141 GB | 990 | ~9,000 |
| L40S 48GB | 48 GB | 362 | ~3,200 |
| RTX 4090 24GB | 24 GB | 330 | ~2,800 |
| RTX 3090 24GB | 24 GB | 142 | ~1,400 |
| RTX 3060 12GB | 12 GB | 38 | ~400 |

These are approximate for Llama-3-7B full fine-tuning with bf16. LoRA will be faster, larger models will be proportionally slower.

See [references/diagnosis-playbook.md](references/diagnosis-playbook.md) for detailed diagnosis procedures with code.
See [references/wandb-patterns.md](references/wandb-patterns.md) for advanced W&B configuration patterns.
