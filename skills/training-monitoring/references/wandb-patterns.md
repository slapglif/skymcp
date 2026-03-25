# W&B (Weights & Biases) Patterns for ML Training

Advanced configuration patterns for training monitoring with Weights & Biases.

## Basic Setup

### Installation and Login

```bash
pip install wandb
wandb login  # Enter API key from wandb.ai/authorize
```

### Environment Variable Auth (SkyPilot)

```yaml
# SkyPilot YAML
envs:
  WANDB_API_KEY: null  # Reads from local environment

setup: |
  pip install wandb

run: |
  python train.py
```

## Initialization Patterns

### Standard Init

```python
import wandb

run = wandb.init(
    project="my-project",          # W&B project name
    entity="my-team",              # W&B team/org
    name="llama3-8b-lora-r32",    # Run name
    group="lora-experiments",      # Group related runs
    tags=["llama3", "lora", "sft"],
    notes="Testing LoRA rank 32 on custom SFT data",
    config={
        "model": "meta-llama/Llama-3-8B",
        "method": "lora",
        "lora_rank": 32,
        "lora_alpha": 64,
        "learning_rate": 2e-4,
        "batch_size": 32,
        "epochs": 3,
        "dataset": "custom-sft-v2",
        "gpu": "A100-80GB",
        "num_gpus": 4,
    },
)
```

### SkyPilot-Aware Init

```python
import os
import wandb

run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "default"),
    name=os.environ.get("SKYPILOT_TASK_ID", f"local-{os.getpid()}"),
    tags=[
        f"gpu-{os.environ.get('SKYPILOT_NUM_GPUS_PER_NODE', '1')}",
        f"nodes-{os.environ.get('SKYPILOT_NUM_NODES', '1')}",
    ],
    config={
        "skypilot_task_id": os.environ.get("SKYPILOT_TASK_ID"),
        "skypilot_node_rank": os.environ.get("SKYPILOT_NODE_RANK"),
        "skypilot_num_nodes": os.environ.get("SKYPILOT_NUM_NODES"),
    },
    resume="allow",  # Resume if SKYPILOT_TASK_ID matches (spot recovery)
)
```

### Resume After Spot Preemption

```python
# SKYPILOT_TASK_ID is stable across preemption
task_id = os.environ.get("SKYPILOT_TASK_ID", "")

run = wandb.init(
    project="my-project",
    id=task_id,       # Use stable ID so preempted runs resume
    resume="allow",   # "allow" = resume if exists, create if not
)
```

## Logging Patterns

### Training Metrics

```python
for step, batch in enumerate(dataloader):
    # Forward pass
    loss = train_step(batch)

    # Compute metrics
    grad_norm = compute_grad_norm(model)
    lr = scheduler.get_last_lr()[0]
    throughput = batch_tokens / step_time

    # Log core metrics
    wandb.log({
        "train/loss": loss.item(),
        "train/learning_rate": lr,
        "train/gradient_norm": grad_norm,
        "train/throughput_tokens_per_sec": throughput,
        "train/global_step": step,
        "train/epoch": epoch + step / len(dataloader),
    }, step=step)

    # Log system metrics periodically
    if step % 50 == 0:
        wandb.log({
            "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "system/gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "system/gpu_utilization": get_gpu_util(),
        }, step=step)
```

### Validation Metrics

```python
if step % eval_interval == 0:
    val_loss, val_metrics = evaluate(model, val_loader)
    wandb.log({
        "val/loss": val_loss,
        "val/perplexity": math.exp(val_loss),
        "val/accuracy": val_metrics["accuracy"],
    }, step=step)
```

### Learning Rate Schedule Visualization

```python
# Log entire LR schedule at init
lr_data = []
temp_scheduler = copy.deepcopy(scheduler)
for s in range(total_steps):
    lr_data.append([s, temp_scheduler.get_last_lr()[0]])
    temp_scheduler.step()

wandb.log({
    "lr_schedule": wandb.plot.line(
        wandb.Table(data=lr_data, columns=["step", "lr"]),
        "step", "lr", title="Learning Rate Schedule"
    )
})
```

### Gradient Histograms

```python
if step % 100 == 0:
    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({
                f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy()),
                f"weights/{name}": wandb.Histogram(param.data.cpu().numpy()),
            }, step=step)
```

### Sample Predictions

```python
if step % 500 == 0:
    model.eval()
    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Explain quantum computing in simple terms:",
    ]
    predictions = []
    for prompt in prompts:
        output = generate(model, tokenizer, prompt, max_tokens=100)
        predictions.append([prompt, output])

    wandb.log({
        "predictions": wandb.Table(
            data=predictions,
            columns=["prompt", "output"]
        )
    }, step=step)
    model.train()
```

## Alerts

### NaN/Inf Detection

```python
if math.isnan(loss.item()) or math.isinf(loss.item()):
    wandb.alert(
        title="NaN/Inf Loss Detected",
        text=f"Loss became {loss.item()} at step {step}. "
             f"LR: {lr:.2e}, Grad norm: {grad_norm:.2f}",
        level=wandb.AlertLevel.ERROR,
        wait_duration=300,  # Don't re-alert for 5 minutes
    )
```

### OOM Warning

```python
gpu_util = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_mem
if gpu_util > 0.95:
    wandb.alert(
        title="GPU Memory Critical",
        text=f"GPU memory at {gpu_util*100:.1f}%. OOM risk.",
        level=wandb.AlertLevel.WARN,
    )
```

### Training Stall Detection

```python
# Detect if loss hasn't improved in N steps
if len(loss_history) > 1000:
    recent_mean = sum(loss_history[-100:]) / 100
    older_mean = sum(loss_history[-1000:-900]) / 100
    if abs(recent_mean - older_mean) < 0.001:
        wandb.alert(
            title="Training Stalled",
            text=f"Loss unchanged for ~900 steps. "
                 f"Recent: {recent_mean:.4f}, Older: {older_mean:.4f}",
            level=wandb.AlertLevel.WARN,
        )
```

## Artifacts

### Save Model Checkpoint

```python
# Save best model as artifact
if val_loss < best_val_loss:
    best_val_loss = val_loss
    artifact = wandb.Artifact(
        name=f"model-{run.id}",
        type="model",
        description=f"Best checkpoint at step {step}, val_loss={val_loss:.4f}",
        metadata={"step": step, "val_loss": val_loss},
    )
    artifact.add_dir("/checkpoints/best/")
    run.log_artifact(artifact)
```

### Save Training Data Hash

```python
# Track data provenance
data_artifact = wandb.Artifact(
    name="training-data",
    type="dataset",
    metadata={
        "num_samples": len(dataset),
        "hash": compute_dataset_hash(dataset),
    },
)
data_artifact.add_reference(f"s3://my-bucket/data/train.parquet")
run.log_artifact(data_artifact)
```

## Sweeps (Hyperparameter Search)

### Define Sweep

```python
sweep_config = {
    "method": "bayes",  # bayes, random, grid
    "metric": {"name": "val/loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"min": 1e-6, "max": 1e-3, "distribution": "log_uniform_values"},
        "lora_rank": {"values": [8, 16, 32, 64]},
        "lora_alpha": {"values": [16, 32, 64, 128]},
        "weight_decay": {"min": 0.001, "max": 0.1},
        "warmup_steps": {"min": 50, "max": 500},
        "batch_size": {"values": [16, 32, 64]},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 100,
        "eta": 3,
    },
}

sweep_id = wandb.sweep(sweep_config, project="my-project")
```

### Run Sweep with SkyPilot

```bash
# Launch sweep agents as SkyPilot managed jobs
for i in $(seq 1 8); do
  sky jobs launch sweep-agent.yaml --env WANDB_SWEEP_ID=$SWEEP_ID --env AGENT_ID=$i
done
```

```yaml
# sweep-agent.yaml
name: sweep-agent-${AGENT_ID}
resources:
  accelerators: A100:1
  use_spot: true

envs:
  WANDB_API_KEY: null
  WANDB_SWEEP_ID: null

run: |
  wandb agent $WANDB_SWEEP_ID
```

## Multi-GPU / Multi-Node Logging

### Log Only from Rank 0

```python
import os

is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0

if is_main_process:
    wandb.init(project="my-project")

# In training loop
if is_main_process:
    wandb.log({"train/loss": loss.item()})
```

### Aggregate Metrics Across Ranks

```python
import torch.distributed as dist

def all_reduce_mean(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor

# Aggregate loss across all GPUs before logging
loss_tensor = torch.tensor(loss.item()).cuda()
avg_loss = all_reduce_mean(loss_tensor)

if is_main_process:
    wandb.log({"train/loss": avg_loss.item()})
```

## Framework Integrations

### HuggingFace Trainer

```python
from transformers import TrainingArguments

args = TrainingArguments(
    report_to="wandb",
    run_name="my-run",
    logging_steps=10,
)
# W&B logging is automatic
```

### Axolotl

```yaml
# In axolotl config YAML
wandb_project: my-project
wandb_entity: my-team
wandb_name: my-run
wandb_log_model: false
logging_steps: 10
```

### NeMo

```python
from nemo.lightning.pytorch.callbacks import WandbLogger

wandb_logger = WandbLogger(
    project="my-project",
    name="my-run",
    save_dir="/logs",
)

trainer = NeMoTrainer(
    logger=wandb_logger,
    log_every_n_steps=10,
)
```

### PyTorch Lightning

```python
from lightning.pytorch.loggers import WandbLogger

logger = WandbLogger(project="my-project", name="my-run")
trainer = Trainer(logger=logger)
```

## Dashboard Organization

### Recommended Panel Layout

Create a W&B dashboard with these sections:

**Row 1: Core Metrics (always visible)**
- train/loss (line chart)
- val/loss (line chart)
- train/learning_rate (line chart)

**Row 2: Training Health**
- train/gradient_norm (line chart)
- train/throughput_tokens_per_sec (line chart)
- system/gpu_memory_allocated_gb (line chart)

**Row 3: System**
- system/gpu_utilization (line chart)
- Custom: step time histogram

**Row 4: Evaluation**
- val/perplexity (line chart)
- predictions table (latest)

### Run Comparison

Use W&B run comparison to compare hyperparameter settings:
1. Select runs in the sidebar
2. Click "Compare"
3. Use parallel coordinates plot to see hyperparameter-to-metric relationships
4. Sort runs by val/loss to find best configuration

## Cleanup

```python
# Always finish the run
wandb.finish()

# In a training script with error handling
try:
    train()
except Exception as e:
    wandb.alert(title="Training Failed", text=str(e), level=wandb.AlertLevel.ERROR)
    raise
finally:
    wandb.finish()
```
