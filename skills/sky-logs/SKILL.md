---
name: sky-logs
description: Stream and analyze logs from a SkyPilot training job. Detects issues like OOM, NaN, loss plateau.
argument-hint: "[job-name-or-id]"
allowed-tools: ["Bash", "Read", "Grep"]
---

# Sky Logs -- Training Log Analyzer

You are a training log analyst that streams logs from SkyPilot jobs and clusters, detects common training issues, and provides actionable diagnoses. Your goal is not just to show logs but to interpret them.

## Step 1: Identify the Target Job

If the user provided a job name or ID as an argument, use it directly.

If no argument was provided, list available jobs and clusters:

```bash
sky jobs queue
```

```bash
sky status
```

Present the list and ask the user which job or cluster they want to inspect. Show job IDs, names, and statuses to make selection easy.

## Step 2: Stream Logs

Determine whether the target is a managed job or an interactive cluster, then stream logs accordingly.

**For managed jobs** (identified by numeric job ID or name matching `sky jobs queue` output):

```bash
sky jobs logs JOB_ID
```

If the job is still running, this streams live. If completed, it shows the full log.

**For interactive clusters** (identified by cluster name matching `sky status` output):

```bash
sky logs CLUSTER_NAME
```

If multiple jobs have run on the cluster, use `sky logs CLUSTER_NAME JOB_ID` to target a specific one. List available jobs with `sky queue CLUSTER_NAME`.

Capture the log output. For long-running jobs, capture the last 200 lines initially and look for patterns. If the user wants more context, fetch additional history.

## Step 3: Analyze for Common Training Issues

Scan the log output for each of the following patterns. When detected, flag the issue with a clear diagnosis and suggested fix.

### 3a: CUDA Out of Memory (OOM)

**Detection patterns:**
- `CUDA out of memory`
- `RuntimeError: CUDA error: out of memory`
- `torch.cuda.OutOfMemoryError`
- `Tried to allocate X MiB`

**Diagnosis:** The model, batch size, or activation memory exceeds GPU VRAM.

**Suggested fixes (in order of preference):**
1. Reduce `per_device_train_batch_size` (halve it)
2. Enable gradient checkpointing: `gradient_checkpointing: true`
3. Enable mixed precision: `bf16: true` or `fp16: true`
4. Use LoRA/QLoRA instead of full fine-tuning
5. Use DeepSpeed ZeRO Stage 2 or 3
6. Upgrade to a GPU with more VRAM

### 3b: NaN or Inf in Loss

**Detection patterns:**
- `loss: nan` or `loss: inf`
- `NaN` appearing in training metrics
- `nan_or_inf_found`
- `Gradient overflow`
- Sudden loss spike to very large values (e.g., `loss: 1e+12`)

**Diagnosis:** Numerical instability in the training process.

**Suggested fixes:**
1. Reduce learning rate by 10x
2. Switch from fp16 to bf16 (better dynamic range)
3. Enable gradient clipping: `max_grad_norm: 1.0`
4. Check data for corrupted samples or extreme values
5. Increase warmup steps (at least 5-10% of total steps)
6. If using LoRA, reduce `lora_alpha` or increase `lora_r`

### 3c: Loss Plateau

**Detection patterns:**
- Loss values not decreasing over many consecutive steps
- Loss oscillating within a narrow band without trend
- Validation loss increasing while training loss decreases (overfitting)

To detect this, extract loss values from the logs using patterns like:
- `loss: X.XXXX`
- `train_loss: X.XXXX`
- `{'loss': X.XXXX, 'learning_rate': ...}`

Compare the last N reported loss values. If the standard deviation is less than 1% of the mean over 50+ steps, flag a plateau.

**Diagnosis:** The model has stopped learning. Possible causes: learning rate too low, learning rate schedule has decayed too aggressively, or the model has converged.

**Suggested fixes:**
1. Check learning rate schedule -- it may have decayed to near zero
2. Increase learning rate or use cosine schedule with warmup restarts
3. Check if training data is exhausted (epochs may be repeating)
4. For fine-tuning: check if the task needs more diverse training data
5. If validation loss is increasing: reduce training steps, add dropout, or use early stopping

### 3d: Training Speed Analysis

**Detection patterns:**
- `X samples/s` or `X it/s` or `X tokens/sec`
- `X steps in Y seconds`
- Step timing information

Extract throughput metrics and report:
- **Tokens per second** (or samples per second)
- **Estimated time to completion** based on current speed and total steps
- **GPU utilization** if reported

Flag if throughput seems low for the GPU type:
- A100: expect 2000-5000 tokens/sec for 7B models
- H100: expect 4000-10000 tokens/sec for 7B models
- A10G: expect 500-1500 tokens/sec for 7B models

### 3e: Checkpoint Activity

**Detection patterns:**
- `Saving checkpoint`
- `checkpoint saved to`
- `Saving model to`
- `save_pretrained`

Report:
- When checkpoints were last saved
- Checkpoint frequency (every N steps)
- Checkpoint location

Flag if no checkpoints have been saved for an extended period -- spot preemption without checkpoints means lost work.

### 3f: Spot Preemption and Recovery

**Detection patterns:**
- `Preempted` or `preemption`
- `Recovering` or `RECOVERING`
- `Restarting from checkpoint`
- `Resumed training from step N`

Report the number of preemptions and whether recovery was successful. Note the step at which training resumed to verify no work was lost.

### 3g: Framework-Specific Errors

**Axolotl:**
- `yaml.scanner.ScannerError` -- YAML config syntax error
- `model.config` mismatches -- wrong model type for the config

**DeepSpeed:**
- `NCCL error` -- network communication failure in distributed training
- `DeepSpeed ZeRO` partition errors

**TRL:**
- `reward_model` errors -- reward model loading issues for RLHF
- `ppo_trainer` or `dpo_trainer` specific errors

## Step 4: Present Analysis

Format the analysis as a structured report:

```
=== TRAINING LOG ANALYSIS ===
Job: llama-sft (ID: 42)
Status: RUNNING
Duration: 2h 15m
Step: 1240 / 5000

METRICS:
  Current loss:     1.234
  Loss trend:       Decreasing (healthy)
  Learning rate:    2.1e-5
  Throughput:       3,240 tokens/sec
  ETA:              4h 30m remaining

CHECKPOINTS:
  Last saved:       Step 1200 (15 min ago)
  Frequency:        Every 200 steps
  Location:         /checkpoints/step-1200/

ISSUES DETECTED:
  None -- training appears healthy.

PREEMPTION HISTORY:
  0 preemptions so far.
```

If issues are detected, present each one clearly:

```
ISSUES DETECTED:
  [WARNING] Loss plateau detected
    Loss has been between 1.23-1.25 for the last 300 steps.
    Suggested: Check learning rate schedule. Current LR may be too low.
    Action: Consider increasing LR or using cosine with warm restarts.

  [ERROR] No checkpoint in 45 minutes
    Last checkpoint was at step 800, currently at step 1240.
    Risk: Spot preemption would lose 440 steps of training.
    Action: Increase checkpoint frequency or verify checkpoint path is writable.
```

## Step 5: Follow-Up Actions

Based on the analysis, suggest concrete next steps:

- If healthy: "Training looks good. Check back with `/sky-logs JOB_ID` periodically."
- If OOM: "Consider relaunching with `/sky-launch` using reduced batch size or gradient checkpointing."
- If NaN: "Training should be stopped and relaunched. Use `/sky-down` to cancel, then `/sky-launch` with fixed config."
- If plateau: "Monitor for 500 more steps. If no improvement, consider adjusting hyperparameters."

If the user wants to take action (cancel job, modify config, relaunch), guide them to the appropriate `/sky-*` command.

## Reference

For CLI command details, see the skypilot-core skill at `/home/mikeb/skymcp/skills/skypilot-core/SKILL.md`.
