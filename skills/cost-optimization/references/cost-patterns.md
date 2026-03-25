# Advanced Cost Optimization Patterns

Beyond basic spot/on-demand selection -- patterns for systematic cost reduction in ML workflows.

## Pattern 1: Progressive GPU Escalation

Start on cheap GPUs, promote to expensive only when justified.

```
Phase 1: Prototype (T4/A10G)
  - Debug data pipeline, verify training loop
  - Cost: $0.15-0.80/hr
  - Duration: hours

Phase 2: Validate (A100 40GB, single GPU)
  - Confirm loss curve shape, learning rate
  - Cost: $1.00-1.50/hr spot
  - Duration: hours to 1 day

Phase 3: Scale (A100 80GB x8 or H100 x8)
  - Full training run
  - Cost: $8-25/hr spot
  - Duration: days

Phase 4: Final eval (A100 single GPU)
  - Drop back to cheap GPU for evaluation
  - Cost: $1.00-1.50/hr spot
  - Duration: hours
```

**SkyPilot implementation:**
```yaml
# phase1-prototype.yaml
resources:
  accelerators: T4:1
  use_spot: true

# phase3-scale.yaml
resources:
  accelerators: H100:8
  use_spot: true
```

**Savings:** 60-80% compared to running everything on H100.

## Pattern 2: Preemption-Aware Checkpointing

Adapt checkpoint frequency based on spot preemption risk.

```python
# In training script
import os
import time

LAST_CHECKPOINT = time.time()
SPOT_DETECTED = os.environ.get("SKYPILOT_SPOT", "false") == "true"

# Shorter interval on spot, longer on on-demand
CHECKPOINT_INTERVAL = 900 if SPOT_DETECTED else 3600  # 15min vs 1hr

def maybe_checkpoint(step, model, optimizer):
    global LAST_CHECKPOINT
    elapsed = time.time() - LAST_CHECKPOINT
    if elapsed >= CHECKPOINT_INTERVAL:
        save_checkpoint(step, model, optimizer)
        LAST_CHECKPOINT = time.time()
```

**Cost impact:** More checkpoints = more I/O cost but less wasted compute on preemption.

**Optimal interval formula:**
```
interval = sqrt(2 * avg_preemption_interval * checkpoint_duration)

Example: avg preemption every 2 hours, checkpoint takes 30 seconds
interval = sqrt(2 * 7200 * 30) = sqrt(432000) ~= 657 seconds ~= 11 minutes
```

## Pattern 3: Time-of-Day Spot Pricing

Spot prices are lower during off-peak hours (nights, weekends in the instance's region).

```bash
# Check current spot pricing
sky gpus --all A100 --region us-east-1

# Launch during off-peak (automate with cron)
# US off-peak: 10 PM - 6 AM EST
# EU off-peak: 10 PM - 6 AM CET
```

**SkyPilot handles this implicitly** -- it checks live prices and picks the cheapest available option at launch time. But you can optimize further:

```bash
# Launch script that waits for cheap spot
MAX_PRICE=1.50  # $/hr per GPU
while true; do
    PRICE=$(sky launch --dryrun job.yaml 2>&1 | grep "Estimated cost" | awk '{print $NF}')
    if (( $(echo "$PRICE < $MAX_PRICE" | bc -l) )); then
        sky launch job.yaml --use-spot --yes
        break
    fi
    echo "Current price $PRICE > $MAX_PRICE, waiting 30 minutes..."
    sleep 1800
done
```

## Pattern 4: Heterogeneous GPU Training

Use cheaper GPUs for some roles in distributed training.

```yaml
# Parameter server on cheap GPU, workers on A100
# (Framework-specific -- works with some distributed setups)
resources:
  any_of:
    - accelerators: A100:8
      use_spot: true
    - accelerators: A100-80GB:8
      use_spot: true
    - accelerators: L40S:8
      use_spot: true
```

**Practical approach:** Let SkyPilot cascade through GPU types. If all A100 spot is taken, L40S might be available and still fast enough.

## Pattern 5: Checkpoint Storage Lifecycle

Manage checkpoint storage costs by promoting/archiving/deleting.

```bash
# Storage cost: ~$0.023/GB/month (S3 Standard)
# A 70B model checkpoint: ~140GB
# 10 checkpoints: 1.4TB = $32/month

# Lifecycle policy: keep only last 3 checkpoints + best
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-checkpoints \
  --lifecycle-configuration '{
    "Rules": [{
      "ID": "delete-old-checkpoints",
      "Prefix": "checkpoints/",
      "Status": "Enabled",
      "Expiration": {"Days": 30},
      "NoncurrentVersionExpiration": {"NoncurrentDays": 7}
    }]
  }'
```

**SkyPilot bucket cleanup:**
```bash
# List all SkyPilot managed storage
sky storage ls

# Delete specific bucket
sky storage delete my-old-bucket

# Clean up all unused storage
sky storage ls | grep "STALE" | awk '{print $1}' | xargs -I {} sky storage delete {} --yes
```

## Pattern 6: Multi-Job Batch Scheduling

Run many small jobs (sweeps, eval) as a batch to minimize cluster setup/teardown overhead.

```yaml
# sweep.yaml -- runs all experiments on one cluster
name: hp-sweep
resources:
  accelerators: A100:1
  use_spot: true

run: |
  for LR in 1e-4 3e-4 1e-3; do
    for BATCH in 16 32 64; do
      echo "Running LR=$LR BATCH=$BATCH"
      python train.py --lr $LR --batch-size $BATCH \
        --max-steps 1000 \
        --output /results/lr${LR}_bs${BATCH}/
    done
  done
```

**vs launching separate clusters:**
- Single cluster: 1 setup (~5 min) + all runs
- Separate clusters: N setups (~5 min each) + billing minimum per cluster
- Savings: avoid N-1 setup costs and per-instance minimum billing

## Pattern 7: Eval on Smallest Possible GPU

Evaluation does not need the same GPU as training. Downsize aggressively.

```
Training:    H100:8 ($20/hr spot)
Eval:        T4:1   ($0.15/hr spot)   -- if model fits in 16GB
             A10G:1 ($0.40/hr spot)   -- if model needs up to 24GB
             A100:1 ($1.00/hr spot)   -- for large models only
```

**For models that need more VRAM than available on cheap GPUs:**
```bash
# Use VLLM with quantization for eval
lm_eval --model vllm \
  --model_args pretrained=/model,dtype=float16,quantization=awq \
  --tasks mmlu \
  --batch_size auto
```

## Pattern 8: Reserved Instance Strategy

For teams with predictable, steady-state usage.

**Decision matrix:**

| Monthly GPU Hours | Recommendation | Typical Savings |
|-------------------|----------------|-----------------|
| < 200 hrs/month | Spot + On-Demand | 40-60% vs on-demand |
| 200-500 hrs/month | 1-Year Reserved + Spot overflow | 50-65% |
| 500+ hrs/month | 3-Year Reserved + Spot overflow | 60-75% |

**Hybrid reserved + spot:**
```
Base load:     Reserved instances (predictable cost)
Burst/overflow: Spot instances (cheap, preemptible)
Emergency:     On-demand (guaranteed, expensive)
```

## Pattern 9: Region Arbitrage

Different regions have different prices and availability. SkyPilot exploits this automatically, but you can optimize further.

```yaml
# Explicit region preferences (cheapest first)
resources:
  accelerators: A100:8
  use_spot: true
  any_of:
    - cloud: gcp
      region: us-central1      # Often cheapest GCP
    - cloud: aws
      region: us-east-2        # Ohio, often cheap
    - cloud: aws
      region: us-west-2        # Oregon, high availability
```

**Data locality caveat:** If your data is in us-east-1, launching in ap-southeast-1 adds data transfer cost and latency. Keep compute near data.

## Pattern 10: Cluster Reuse

Reuse clusters across jobs to avoid setup overhead.

```bash
# Launch cluster once
sky launch -c my-cluster setup.yaml

# Run multiple jobs on same cluster
sky exec my-cluster job1.yaml
sky exec my-cluster job2.yaml
sky exec my-cluster job3.yaml

# Tear down when done
sky down my-cluster
```

**Savings:** Avoid 3-10 minutes of setup per job (instance provisioning, dependency installation). For many short jobs, this adds up significantly.

## Cost Tracking Dashboard

```bash
# Daily cost check (add to crontab or alias)
alias sky-costs='echo "=== Running Clusters ===" && sky status && echo "" && echo "=== Cost Report ===" && sky cost-report && echo "" && echo "=== Storage ===" && sky storage ls'
```

**Alerting pattern:**
```bash
# In crontab: check every hour, alert if daily spend exceeds $X
0 * * * * /path/to/check_budget.sh

# check_budget.sh
#!/bin/bash
DAILY_BUDGET=200
CURRENT=$(sky cost-report --json 2>/dev/null | python3 -c "
import json,sys
data = json.load(sys.stdin)
print(sum(c.get('cost',0) for c in data.get('clusters',[])))
" 2>/dev/null || echo "0")

if (( $(echo "$CURRENT > $DAILY_BUDGET" | bc -l) )); then
  echo "BUDGET ALERT: Daily spend $CURRENT exceeds $DAILY_BUDGET" | \
    mail -s "SkyPilot Budget Alert" team@example.com
fi
```

## Summary: Cost Optimization Checklist

- [ ] Always use `--dryrun` before launching
- [ ] Default to spot for all training and eval
- [ ] Set auto-stop on every cluster (no exceptions)
- [ ] Use mixed failover (spot across clouds, on-demand fallback)
- [ ] Start on cheap GPUs, escalate only when validated
- [ ] Checkpoint frequently on spot instances
- [ ] Clean up old checkpoints and storage
- [ ] Downsize GPU for eval (T4/A10G if model fits)
- [ ] Batch small jobs to avoid setup overhead
- [ ] Track spend daily
