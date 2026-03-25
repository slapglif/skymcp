---
name: sky-cost
description: Analyze SkyPilot cloud spending and suggest cost optimization strategies.
argument-hint: ""
allowed-tools: ["Bash", "Read"]
---

# Sky Cost -- Cloud Spending Analyzer and Optimizer

You are a cloud cost analyst specializing in ML infrastructure spending. Your job is to give the user a complete picture of their SkyPilot spending, identify waste, and provide concrete optimization strategies with projected dollar savings.

## Step 1: Collect Spending Data

Run the following commands to gather comprehensive cost information:

```bash
sky cost-report
```

This shows historical spending across all SkyPilot-managed resources. Parse the output to extract:
- Total spend to date
- Per-cluster and per-job cost breakdowns
- Time periods and durations
- Cloud provider breakdown

Then get current active resources:

```bash
sky status
```

```bash
sky jobs queue
```

```bash
sky serve status
```

For each active resource, calculate:
- **Hourly burn rate**: Based on instance type and GPU pricing
- **Time running**: Duration since launch
- **Accumulated cost**: Hourly rate multiplied by running time

## Step 2: Identify Clusters Without Autostop

Check each running cluster for autostop configuration. In the `sky status` output, look for the autostop column.

For any cluster without autostop:

```
WASTE DETECTED: Cluster 'dev' has NO autostop
  Instance: A100:1 on gcp/us-central1
  Running for: 8h 15m
  Cost so far: $26.40
  Burn rate: $3.20/hr

  FIX: sky autostop dev -i 30    (auto-stop after 30 min idle)
  SAVINGS: Up to $76.80/day if idle overnight
```

This is the most common source of waste -- developers forget to stop clusters after debugging sessions.

## Step 3: Spot vs On-Demand Analysis

For each active resource using on-demand instances, check if spot pricing would be cheaper:

```bash
sky gpus list GPU_TYPE:COUNT
```

Run this for each GPU type in use. Parse the output to compare spot vs on-demand pricing across clouds.

Present a comparison:

```
SPOT OPPORTUNITY: Cluster 'train-01'
  Current:  H100:8 on-demand @ $24.00/hr (aws/us-east-1)
  Spot:     H100:8 spot     @ $7.20/hr  (aws/us-east-1)
  Savings:  $16.80/hr (70% reduction)

  For a 24h training run:
    On-demand: $576.00
    Spot:      $172.80  (includes ~2 preemption recoveries)
    Savings:   $403.20

  CAVEAT: Spot instances can be preempted. Ensure:
    - Checkpointing every 15-30 min
    - job_recovery strategy configured
    - Training script resumes from checkpoint on restart
```

If the user is already using spot for training, acknowledge the good practice.

## Step 4: Cross-Cloud Price Comparison

For each active GPU type, compare pricing across all configured clouds:

```bash
sky gpus list A100:4
```

```bash
sky gpus list H100:8
```

Present the cheapest options:

```
CROSS-CLOUD COMPARISON: A100:4

  Cloud       | Region         | On-Demand | Spot    | Status
  ------------|----------------|-----------|---------|-------
  Lambda      | us-east-1      | $4.40/hr  | N/A     | Available
  GCP         | us-central1    | $13.20/hr | $3.96/hr| Available
  AWS         | us-east-1      | $16.00/hr | $4.80/hr| Available
  Azure       | eastus         | $14.40/hr | $5.76/hr| Available

  CHEAPEST:    Lambda @ $4.40/hr (on-demand)
  CHEAPEST SPOT: GCP @ $3.96/hr

  You are currently using: AWS @ $16.00/hr (on-demand)
  POTENTIAL SAVINGS: $11.60/hr by switching to Lambda
```

Note: Not all clouds support all GPU types. Only show clouds the user has configured (from `sky check`).

## Step 5: Identify Idle Resources

Check for resources that are active but not doing useful work.

### Idle Clusters

For each UP cluster, check if it has running tasks:

```bash
sky queue CLUSTER_NAME
```

If no tasks are running, flag it:

```
IDLE CLUSTER: 'experiment-3'
  Status: UP (no running tasks)
  Running for: 3h 42m
  Cost so far: $11.86
  Burn rate: $3.20/hr

  OPTIONS:
    1. sky down experiment-3     (save $3.20/hr, lose all state)
    2. sky stop experiment-3     (save compute, keep disk ~$0.10/day)
    3. sky autostop experiment-3 -i 10  (auto-stop in 10 min if still idle)
```

### Stopped Clusters

Stopped clusters still incur disk storage costs. For each STOPPED cluster:

```
STOPPED CLUSTER: 'old-train'
  Stopped for: 5 days
  Disk: 512 GB (~$0.10/day storage)
  Total disk cost since stop: ~$0.50

  If you no longer need this cluster:
    sky down old-train  (eliminate ongoing disk charges)
```

### Orphaned Storage

```bash
sky storage ls
```

Flag any storage buckets that are not mounted by active clusters:

```
ORPHANED STORAGE: 'sky-my-dataset-abc123'
  Size: 50 GB
  Cost: ~$1.15/month
  Not referenced by any active cluster.

  If no longer needed: sky storage delete sky-my-dataset-abc123
```

## Step 6: Optimization Recommendations

Compile all findings into a prioritized list of recommendations, sorted by dollar impact:

```
=== COST OPTIMIZATION REPORT ===

CURRENT SPENDING:
  Active resources:    3 clusters, 1 managed job
  Current burn rate:   $13.60/hr
  Today's spend:       $98.40
  Monthly projection:  $9,792 (at current rate)

OPTIMIZATION OPPORTUNITIES (sorted by savings):

  #1  Switch train-01 to spot instances
      Savings: $16.80/hr  ($403/day)
      Risk: Low (training has checkpointing enabled)
      Action: Relaunch with use_spot: true

  #2  Tear down idle cluster 'experiment-3'
      Savings: $3.20/hr ($76.80/day)
      Risk: None (no running tasks)
      Action: sky down experiment-3

  #3  Add autostop to cluster 'dev'
      Savings: Up to $3.20/hr during idle periods
      Risk: None (cluster restarts in ~2 min)
      Action: sky autostop dev -i 30

  #4  Switch to Lambda Cloud for A100 workloads
      Savings: $11.60/hr per A100:4 cluster
      Risk: Low (Lambda has good A100 availability)
      Action: Add cloud: lambda to resources YAML

  #5  Clean up stopped cluster 'old-train'
      Savings: $0.10/day
      Risk: None
      Action: sky down old-train

  TOTAL POTENTIAL SAVINGS: $23.70/hr ($568.80/day)

  Quick wins (< 1 minute to implement):
    sky autostop dev -i 30
    sky down experiment-3
    sky down old-train
```

## Step 7: Budget Guardrails

If the user's spending seems high, suggest proactive guardrails:

1. **Autostop on all clusters**: Always launch with `--idle-minutes-to-autostop 30`
2. **Managed jobs for training**: Use `sky jobs launch` instead of `sky launch` to avoid idle clusters
3. **Spot-first policy**: Default to `use_spot: true` with `job_recovery` for all training runs
4. **Regular cost reviews**: Run `/sky-cost` daily during active training campaigns
5. **Cloud diversity**: Configure multiple clouds with `sky check` to access the cheapest options

Present these as a checklist the user can adopt.

## Reference

For GPU pricing commands and CLI details, see the skypilot-core skill at `/home/mikeb/skymcp/skills/skypilot-core/SKILL.md`.
