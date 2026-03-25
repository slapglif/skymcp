---
name: cost-optimization
description: Use when reducing cloud GPU costs, choosing spot vs on-demand, estimating training expenses, managing budgets, configuring auto-stop, or optimizing spend across clouds - covers SkyPilot optimizer, spot failover, pricing tiers, and budget management
---

# Cloud Cost Optimization

## Overview

Minimize cloud GPU spend without sacrificing reliability. SkyPilot's optimizer automatically finds the cheapest available instance across all configured clouds, and its spot/failover system provides 3-6x savings with automatic recovery.

**Core principle:** Never pay on-demand prices for interruptible workloads. Use spot instances with checkpointing and failover, and auto-stop everything.

## When to Use

- Launching any GPU job (training, eval, inference)
- Estimating cost before committing to a training run
- Budget is constrained and you need to reduce spend
- Choosing between spot and on-demand
- Preventing idle cluster waste
- Comparing GPU pricing across clouds

**Do not use for:**
- Latency-critical serving (use cost-optimization only for batch/training)
- Jobs under 5 minutes (spot preemption overhead not worth it)

## SkyPilot Optimizer

SkyPilot automatically ranks all (cloud, region, instance) triples by hourly price and selects the cheapest available option.

```yaml
# Let SkyPilot pick the cheapest cloud/region
resources:
  accelerators: A100:1
  # No cloud specified = search all configured clouds
```

```bash
# See what SkyPilot would pick and the price
sky launch --dryrun job.yaml

# List pricing for a specific GPU across all clouds
sky gpus --all A100:8

# Check actual spend
sky cost-report
```

## Spot Instances: 3-6x Savings

Spot instances are preemptible but dramatically cheaper. SkyPilot handles preemption recovery automatically.

```yaml
resources:
  accelerators: H100:8
  use_spot: true
```

**When to use spot:**
- Training with checkpointing (can resume after preemption)
- Evaluation jobs (stateless, can restart)
- Hyperparameter sweeps (many short jobs)
- Any job that writes progress to persistent storage

**When NOT to use spot:**
- Live serving endpoints (use on-demand or reserved)
- Jobs under 5 minutes (recovery overhead)
- Demos or time-critical deadlines

## Mixed Failover Pattern

The most cost-effective pattern: try spot across clouds, fall back to on-demand only if all spot is unavailable.

```yaml
resources:
  any_of:
    # Try spot on cheapest clouds first
    - accelerators: H100:8
      use_spot: true
      cloud: lambda
    - accelerators: H100:8
      use_spot: true
      cloud: aws
    - accelerators: H100:8
      use_spot: true
      cloud: gcp
    # On-demand fallback (guaranteed availability)
    - accelerators: H100:8
      use_spot: false
      cloud: lambda
```

**Multi-GPU fallover (different GPU types):**
```yaml
resources:
  any_of:
    - accelerators: H100:8
      use_spot: true
    - accelerators: A100-80GB:8
      use_spot: true
    - accelerators: A100:8
      use_spot: true
    - accelerators: H100:8
      use_spot: false     # on-demand fallback
```

## Auto-Stop: Prevent Waste

The single most impactful cost optimization. Idle clusters are pure waste.

```bash
# Stop cluster after 30 minutes idle
sky autostop mycluster -i 30

# Tear down cluster after 30 minutes idle (no storage cost)
sky autostop mycluster -i 30 --down

# Set auto-stop at launch time
sky launch job.yaml --idle-minutes-to-autostop 30

# Set auto-stop with teardown at launch time
sky launch job.yaml --idle-minutes-to-autostop 30 --down
```

**In YAML:**
```yaml
resources:
  accelerators: A100:1
  # Auto-stop is not in YAML -- pass via CLI flags
```

**Rules of thumb:**
- Training jobs: `--idle-minutes-to-autostop 10 --down` (tear down when done)
- Development clusters: `--idle-minutes-to-autostop 30` (stop but keep disk)
- Batch eval: `--idle-minutes-to-autostop 5 --down` (tear down fast)

## Cost Estimation

Always estimate before launching long runs.

```bash
# Dry run shows instance selection and hourly price
sky launch --dryrun job.yaml

# List all GPU options with pricing
sky gpus --all

# Filter by GPU type
sky gpus --all H100

# Estimate total cost for a training run
# Formula: hourly_rate * estimated_hours * (1 + overhead_buffer)
# Example: $2.50/hr * 24hr * 1.1 = $66
```

**Quick cost calculator:**
```
Total cost = hourly_rate * hours * 1.1 (10% buffer for setup/teardown)

Spot savings = on_demand_rate * 0.25 to 0.35 (spot is 65-75% cheaper)

Multi-node cost = per_node_rate * num_nodes * hours * 1.1
```

## GPU Pricing Tiers (Approximate, 2026)

For detailed per-cloud pricing, see `references/pricing-guide.md`.

| GPU | Spot ($/hr) | On-Demand ($/hr) | Best For |
|-----|-------------|-------------------|----------|
| H100 80GB | $2.00-4.00 | $8.00-12.00 | Large model training, fastest throughput |
| A100 80GB | $1.00-2.00 | $4.00-6.00 | Standard training, good price/performance |
| A100 40GB | $0.80-1.50 | $3.00-5.00 | Models up to ~30B, inference |
| L40S 48GB | $0.80-1.50 | $2.50-4.00 | Training + inference hybrid |
| A10G 24GB | $0.40-0.80 | $1.50-2.50 | Fine-tuning, small models, eval |
| T4 16GB | $0.15-0.35 | $0.50-1.00 | Inference, small fine-tuning |

**Cloud-specific cost advantages:**
- **Lambda Cloud:** Often cheapest for H100/A100 on-demand
- **AWS:** Best spot availability, largest region selection
- **GCP:** Good A100 spot pricing, TPU option
- **RunPod/Vast.ai:** Cheapest for short bursts (via SkyPilot)

## Checkpointing Frequency vs Cost

More frequent checkpoints = less wasted work on preemption, but more I/O overhead.

```
Optimal checkpoint interval = sqrt(2 * mean_time_between_preemptions * checkpoint_cost)
```

**Rules of thumb:**
- Spot on AWS/GCP: checkpoint every 15-30 minutes
- Spot on Lambda: checkpoint every 30-60 minutes (less preemption)
- On-demand: checkpoint every 1-2 hours (insurance only)
- Async checkpointing: free to checkpoint more often (no training stall)

## Budget Management

**Daily cost cap pattern:**

```bash
# Check spend before launching
sky cost-report

# Launch with awareness of cumulative cost
# No built-in budget cap in SkyPilot -- implement externally:
DAILY_BUDGET=100
CURRENT_SPEND=$(sky cost-report --json | python3 -c "import json,sys; print(sum(j['cost'] for j in json.load(sys.stdin)['clusters']))")
if (( $(echo "$CURRENT_SPEND > $DAILY_BUDGET" | bc -l) )); then
  echo "BUDGET EXCEEDED: $CURRENT_SPEND > $DAILY_BUDGET"
  exit 1
fi
sky launch job.yaml
```

**Cost-aware experiment design:**
- Run fast sanity checks on cheap GPUs (T4/A10G) before full training on H100
- Use `--limit 100` for eval debugging, full eval only for final checkpoints
- Sweep hyperparameters with short runs (1000 steps) before full training
- Use mixed failover to catch cheap spot when available

For advanced cost patterns (preemption-aware scheduling, multi-job orchestration, reserved instance strategy), see `references/cost-patterns.md`.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Forgetting auto-stop | Always pass `--idle-minutes-to-autostop`. Set a shell alias. |
| On-demand for training | Use spot. Checkpointing makes preemption free. |
| Not using `--dryrun` | Always check price before launching. |
| Single-cloud lock-in | Configure multiple clouds. SkyPilot finds cheapest. |
| No checkpointing with spot | Preemption = lost work. Always checkpoint. |
| Over-provisioning GPU | Use smallest GPU that fits your model. A10G for 7B fine-tune, not H100. |
| Forgetting `--down` | Stopped clusters still cost for disk. Use `--down` for batch jobs. |

## Quick Reference

```bash
# See cheapest option for your job
sky launch --dryrun job.yaml

# Launch on cheapest spot
sky launch job.yaml --use-spot --idle-minutes-to-autostop 10 --down

# List running clusters and their costs
sky status

# Cost report
sky cost-report

# Tear down all clusters
sky down --all

# GPU pricing
sky gpus --all H100
```
