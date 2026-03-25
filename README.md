# SkyMCP — SkyPilot ML Training Ecosystem Plugin

End-to-end ML training platform for Claude Code. Launch, monitor, fix, iterate, ablate, and serve models on any cloud GPU using SkyPilot with production-grade frameworks.

## What It Does

Once installed, this plugin transforms Claude Code into a senior ML engineer + MLOps specialist + cloud cost optimizer:

- **Launch training jobs** on any cloud GPU (AWS, GCP, Azure, Lambda, RunPod, 20+ providers)
- **Pick the right framework** automatically (NeMo, Axolotl, torchtune, TRL, DeepSpeed, Megatron)
- **Monitor training** with W&B/TensorBoard integration, automatic issue detection
- **Diagnose and fix** OOM, NaN, loss plateaus, gradient explosion, slow throughput
- **Run experiments** — hyperparameter sweeps, architecture ablations, scaling law studies
- **Optimize costs** — spot instances, multi-cloud failover, autostop, budget management
- **Evaluate models** with lm-evaluation-harness across standard benchmarks
- **Deploy models** with vLLM on SkyServe with autoscaling
- **Real data pipelines** — NeMo Curator, FineWeb, DCLM patterns for production data curation

## Prerequisites

- [SkyPilot](https://github.com/skypilot-org/skypilot) installed and configured (`pip install skypilot[aws,gcp]`)
- Cloud credentials configured (`sky check`)
- Node.js 18+ (for MCP server and hooks)

## Installation

```bash
# Test locally
claude --plugin-dir /path/to/skymcp

# Or copy to your project
cp -r skymcp/.claude-plugin your-project/
```

## Components

### Slash Commands (9)

| Command | Description |
|---------|------------|
| `/sky-launch` | Generate YAML, validate, estimate cost, launch training |
| `/sky-status` | Dashboard of all clusters, jobs, services, costs |
| `/sky-logs` | Stream and analyze logs, detect issues |
| `/sky-down` | Safe teardown with cost savings report |
| `/sky-cost` | Spending analysis and optimization suggestions |
| `/sky-sweep` | Launch hyperparameter sweep across cloud GPUs |
| `/sky-eval` | Run model evaluation benchmarks |
| `/sky-serve` | Deploy model with vLLM + SkyServe autoscaling |
| `/sky-recipe` | Generate end-to-end pipeline (data + train + eval + serve) |

### Auto-Activating Skills (8)

Activate automatically based on conversation context:

- **skypilot-core** — CLI reference, YAML spec, env vars, 21 gotchas
- **ml-training-frameworks** — NeMo vs Axolotl vs torchtune vs TRL decision matrix
- **data-pipeline-design** — NeMo Curator, FineWeb, dedup, quality filtering
- **training-monitoring** — W&B, TensorBoard, OOM/NaN/plateau diagnosis
- **model-evaluation** — lm-eval-harness, lighteval, benchmark selection
- **cost-optimization** — Spot strategies, multi-cloud failover, budget management
- **checkpoint-management** — Distributed checkpoints, LoRA merging, GGUF conversion
- **distributed-training** — Multi-node, DeepSpeed ZeRO, FSDP2, InfiniBand

### Agents (5)

| Agent | Role |
|-------|------|
| **training-orchestrator** | Full lifecycle: framework selection, launch, monitor, recover, iterate |
| **experiment-scientist** | Ablation design, scaling laws, sweep comparison |
| **config-validator** | YAML validation, gotcha detection, cost estimation |
| **training-doctor** | Diagnose OOM, NaN, plateau, divergence, slow throughput |
| **cloud-optimizer** | Spending analysis, spot migration, savings recommendations |

### Hooks (3)

- **PostToolUse** — Captures `sky launch` output, tracks job IDs
- **SessionStart** — Loads active clusters/jobs/costs into context
- **PreToolUse** — Validates destructive ops, suggests spot/autostop

### MCP Server (7 tools)

`sky_status`, `sky_launch`, `sky_logs`, `sky_down`, `sky_cost`, `sky_gpus`, `sky_check`

### Recipe Templates (7)

Ready-to-use SkyPilot YAML recipes in `references/recipes/`:

- `nemo-pretraining.yaml` — Multi-node NeMo 2.0 on H100 cluster
- `axolotl-finetune.yaml` — QLoRA fine-tuning with Axolotl
- `torchtune-finetune.yaml` — Full fine-tuning with torch.compile
- `trl-dpo.yaml` — DPO preference alignment
- `vllm-serve.yaml` — Production inference with SkyServe
- `nemo-curator.yaml` — GPU-accelerated data curation
- `full-pipeline.yaml` — 4-stage: data prep + train + eval + serve

## Quick Start

```bash
# Check cloud credentials
sky check

# Launch a fine-tuning job
# (use /sky-launch in Claude Code for interactive workflow)
sky jobs launch references/recipes/axolotl-finetune.yaml \
  --env HF_TOKEN=$HF_TOKEN \
  --env WANDB_API_KEY=$WANDB_API_KEY

# Monitor
sky jobs queue
sky jobs logs JOB_ID

# Evaluate
sky jobs launch references/recipes/eval.yaml

# Serve
sky serve up references/recipes/vllm-serve.yaml -n my-model
```

## Architecture

```
skymcp/
├── .claude-plugin/plugin.json    # Plugin manifest + MCP config
├── skills/                        # 17 skills (8 auto + 9 commands)
│   ├── skypilot-core/            # Core SkyPilot reference
│   ├── ml-training-frameworks/   # Framework selection
│   ├── sky-launch/               # /sky-launch command
│   └── ...
├── agents/                        # 5 autonomous agents
├── hooks/                         # 3 event hooks + scripts
├── mcp/                          # MCP server (TypeScript)
└── references/recipes/           # 7 ready-to-use YAML templates
```

## License

MIT
