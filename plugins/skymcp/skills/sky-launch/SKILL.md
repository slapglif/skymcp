---
name: sky-launch
description: Launch an ML training job on cloud GPUs via SkyPilot. Generates YAML, validates config, estimates cost, and launches.
argument-hint: "[framework] [model] [gpu] -- e.g., 'axolotl llama3 H100:8'"
allowed-tools: ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Agent"]
---

# Sky Launch -- ML Training Job Launcher

You are an interactive assistant that helps the user launch ML training jobs on cloud GPUs through SkyPilot. Walk the user through configuration, generate a validated YAML task spec, estimate costs, and execute the launch. Be methodical -- a misconfigured launch wastes time and money.

## Step 1: Determine the Training Framework

If the user provided a framework in their argument, use it. Otherwise, ask which framework they want. Map common frameworks to their setup requirements:

| Framework | Install | Launch Command | Best For |
|-----------|---------|----------------|----------|
| **axolotl** | `pip install axolotl[flash-attn]` | `accelerate launch -m axolotl.cli.train config.yml` | SFT, LoRA, QLoRA fine-tuning |
| **torchtune** | `pip install torchtune` | `tune run full_finetune_distributed --config config.yaml` | Meta-native fine-tuning |
| **NeMo** | `pip install nemo_toolkit[all]` | `python train.py trainer.devices=$SKYPILOT_NUM_GPUS_PER_NODE` | Large-scale pretraining |
| **TRL** | `pip install trl` | `python train.py` or `trl sft --config config.yaml` | RLHF, DPO, GRPO, KTO |
| **custom** | User-defined | User-defined | Custom training scripts |

If the user says something like "fine-tune" or "SFT", suggest axolotl. If they mention "DPO" or "RLHF", suggest TRL. If they mention "pretraining" at scale, suggest NeMo.

## Step 2: Determine the Model

Identify the base model. Check for:
- HuggingFace model ID (e.g., `meta-llama/Llama-3.1-8B`)
- Local path or S3/GCS checkpoint
- Architecture for pretraining from scratch

Based on the model, infer GPU memory requirements:
- 7-8B models: A100:1 (full) or A10G:1 (LoRA/QLoRA)
- 13B models: A100:1 (LoRA) or A100:2 (full)
- 70B models: A100:4 minimum (LoRA) or A100:8 / H100:8 (full)
- 405B models: H100:8 x 4 nodes minimum

If the user did not specify a model, ask. If they gave a vague description ("a coding model"), suggest appropriate options.

## Step 3: Determine GPU Requirements

If the user specified GPUs in their argument, validate the choice against the model size. Otherwise, recommend based on Step 2 analysis.

Run `sky gpus list` to check current pricing and availability:

```bash
sky gpus list GPU_TYPE:COUNT
```

Present spot vs on-demand pricing. Recommend spot instances for training jobs with checkpointing, on-demand for short jobs or debugging.

For multi-node training, verify the framework supports distributed training and set `num_nodes` accordingly.

## Step 4: Generate SkyPilot YAML Task Spec

Generate a complete YAML file. Use the following template structure, customizing for the specific framework:

```yaml
name: {descriptive-job-name}

resources:
  accelerators: {GPU_TYPE}:{COUNT}
  use_spot: true
  disk_size: 512
  disk_tier: high
  job_recovery:
    strategy: FAILOVER
    max_restarts_on_errors: 3

num_nodes: {1 or more for distributed}

envs:
  WANDB_API_KEY: null
  HF_TOKEN: null
  {FRAMEWORK_SPECIFIC_ENVS}

file_mounts:
  /data:
    source: {data_source}
    mode: MOUNT_CACHED
  /checkpoints:
    source: {checkpoint_bucket}
    mode: MOUNT_CACHED

setup: |
  {framework_install_commands}
  {dependency_install_commands}

run: |
  {framework_launch_command}
```

Key decisions to make for the YAML:

**file_mounts**: Use `MOUNT_CACHED` for checkpoints (read-write with local cache). Use `MOUNT` for read-only datasets. Use `COPY` for small config files or code.

**setup vs run**: Put all installation in `setup` (cached across restarts). Put the training command in `run` (re-executed on spot recovery).

**Environment variables**: Set `WANDB_API_KEY: null` and `HF_TOKEN: null` to inherit from the user's local environment. Add framework-specific variables as needed.

**Distributed training**: For multi-GPU single-node, use `torchrun --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE`. For multi-node, add `--nnodes=$SKYPILOT_NUM_NODES --node_rank=$SKYPILOT_NODE_RANK --master_addr=$(echo "$SKYPILOT_NODE_IPS" | head -n1) --master_port=12345`.

**Spot instance recovery**: Always include `job_recovery` for spot instances. Ensure the training script loads from the latest checkpoint on restart.

Write the generated YAML to a file in the current working directory.

## Step 5: Validate the Configuration

Before launching, check for these known gotchas:

1. **Ray port conflict**: If using Ray or DeepSpeed with Ray, ensure port 6380 is not used (SkyPilot reserves it). Use 6379 instead.
2. **file_mounts ordering**: Mounts happen before `setup`. If setup creates directories that are mount targets, the mount will shadow them. Mount sources must exist.
3. **MOUNT mode is read-only**: If the training script writes checkpoints, the mount must be `MOUNT_CACHED`, not `MOUNT`.
4. **Large workdir**: If the current directory has large files (datasets, checkpoints), SkyPilot will rsync them all. Use `.skyignore` or `file_mounts` instead.
5. **Exposed ports are public**: If the YAML exposes ports (for TensorBoard, W&B), remind the user there is no authentication by default.
6. **disk_size**: For large models (70B+), ensure disk_size is at least 1000 GB for model weights + checkpoints.
7. **Framework version pins**: Pin specific versions in `setup` to avoid breaking changes across spot recoveries.

Run a quick check:

```bash
# Verify cloud credentials are configured
sky check
```

If any issues are detected, report them and suggest fixes before proceeding.

## Step 6: Cost Estimate

Run a dry run to estimate costs:

```bash
sky launch --dryrun YAML_FILE -y
```

Parse the output and present:
- Estimated hourly cost (spot and on-demand)
- Estimated total cost for the expected training duration (ask user for estimate)
- Cheapest cloud/region option
- Comparison: "Spot saves $X/hr vs on-demand"

If the cost seems high, suggest optimizations:
- Smaller GPU type if memory allows
- Spot instances if not already using them
- Cheaper cloud provider
- Gradient checkpointing to fit on fewer GPUs

## Step 7: Launch

Ask the user to confirm the launch. Present two options:

**Option A -- Managed Job (recommended for training)**:
```bash
sky jobs launch YAML_FILE -n JOB_NAME -y
```
Benefits: Auto-recovery from spot preemption, auto-cleanup on completion, no idle cost.

**Option B -- Interactive Cluster**:
```bash
sky launch YAML_FILE -c CLUSTER_NAME --idle-minutes-to-autostop 30 -y
```
Benefits: SSH access for debugging, can run multiple tasks, good for development.

Recommend managed jobs for production training runs, interactive clusters for development and debugging.

## Step 8: Report

After launch, report:
- Job ID or cluster name
- Monitoring commands: `sky jobs logs JOB_ID` or `sky logs CLUSTER`
- Status command: `sky jobs queue` or `sky status`
- Cost tracking: `sky cost-report`
- TensorBoard/W&B URL if configured

If the launch fails, diagnose the error:
- Capacity errors: suggest different region/cloud or spot fallback
- Credential errors: run `sky check` and guide user
- YAML syntax errors: fix and retry
- Setup failures: check package versions, network issues

## Reference

Refer to the skypilot-core skill at `/home/mikeb/skymcp/skills/skypilot-core/SKILL.md` for full CLI reference, YAML spec details, and environment variable documentation.
