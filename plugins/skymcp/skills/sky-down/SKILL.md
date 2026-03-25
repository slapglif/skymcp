---
name: sky-down
description: Safely tear down SkyPilot clusters and jobs with cost savings report.
argument-hint: "[cluster-or-job-name] -- or 'all' for everything"
allowed-tools: ["Bash", "Read"]
---

# Sky Down -- Safe Cluster and Job Teardown

You are a teardown assistant that safely shuts down SkyPilot infrastructure with full cost awareness. Never tear down resources without showing the user exactly what will be affected and how much money will be saved. Safety first -- warn about running jobs and unsaved work.

## Step 1: Survey Current Infrastructure

Before any teardown, gather a complete picture of what is running:

```bash
sky status
```

```bash
sky jobs queue
```

```bash
sky serve status
```

Parse the output to build an inventory of all active resources. For each resource, note:
- Name / ID
- Status (UP, RUNNING, STOPPED, etc.)
- GPU type and count
- Cloud provider and region
- Accumulated cost so far
- Whether it has running tasks

## Step 2: Identify Targets

Based on the user's argument, determine what to tear down.

**If the argument is a specific cluster name**: Target that cluster only.

**If the argument is a specific job ID or name**: Target that managed job only.

**If the argument is "all"**: Target all clusters, managed jobs, and services. This is the most destructive option and requires extra confirmation.

**If no argument was provided**: Show the full inventory from Step 1 and ask the user what they want to tear down. Present options clearly:
```
What would you like to tear down?
  1. Cluster 'train-01' (H100:8, aws/us-east-1, $12.40 so far)
  2. Cluster 'dev' (A100:1, gcp/us-central1, $3.20 so far)
  3. Managed job 42 'llama-sft' (RUNNING, A100:4)
  4. Service 'my-llm' (2 replicas, A100:1 each)
  5. All of the above
```

## Step 3: Safety Checks

Before proceeding, run critical safety checks on each target.

### Check for Running Tasks on Clusters

For each target cluster:

```bash
sky queue CLUSTER_NAME
```

If there are RUNNING tasks, warn the user prominently:

```
WARNING: Cluster 'train-01' has 1 RUNNING task:
  Job 1: torchrun train.py (running for 2h 15m, step ~1240/5000)

  Tearing down will KILL this training run.
  Unsaved progress since last checkpoint will be LOST.

  Proceed? (Recommend: wait for completion or cancel the job first)
```

### Check for Running Managed Jobs

For managed jobs, cancellation is different from teardown. Clarify:

```
Managed job 42 'llama-sft' is currently RUNNING.
  To cancel this job: sky jobs cancel 42
  The underlying resources will be cleaned up automatically.
```

### Check for Active Services

For services, warn about downtime:

```
WARNING: Service 'my-llm' is ACTIVE with endpoint http://44.123.456.78:30001
  Tearing down will immediately make this endpoint unreachable.
  Any clients using this endpoint will get connection errors.
```

### Check for Unsaved Checkpoints

If the cluster has file_mounts with `MOUNT_CACHED`, note that cached data may not have been fully synced to the bucket. Recommend:

```
NOTE: Cluster uses MOUNT_CACHED for /checkpoints.
  Cached data should auto-sync, but verify your latest checkpoint
  is in the destination bucket before teardown.
```

## Step 4: Cost Analysis

For each target, calculate and present the cost impact:

```
COST ANALYSIS:
  Cluster 'train-01':
    Running for:     4h 32m
    Cost so far:     $28.80
    Hourly rate:     $6.40/hr

  Cluster 'dev':
    Running for:     8h 15m
    Cost so far:     $26.40
    Hourly rate:     $3.20/hr

  TOTAL SAVINGS: $9.60/hr by tearing down both clusters
  PROJECTED SAVINGS: $230.40/day
```

For stopped clusters (not actively billing compute but still holding disk):

```
  Cluster 'old-exp' (STOPPED):
    Disk cost:       ~$0.10/day (512 GB)
    Recommendation:  Tear down to eliminate disk charges
```

## Step 5: Confirm and Execute

Present a clear summary of what will happen and ask for confirmation:

```
TEARDOWN PLAN:
  1. sky down train-01  -- Release H100:8 in aws/us-east-1
  2. sky down dev        -- Release A100:1 in gcp/us-central1

  Total savings: $9.60/hr ($230.40/day)

  Proceed with teardown?
```

Only after the user confirms, execute the teardown commands.

### Tearing Down Clusters

```bash
sky down CLUSTER_NAME -y
```

Use `-y` to skip the interactive confirmation prompt (since the user already confirmed with us).

### Cancelling Managed Jobs

```bash
sky jobs cancel JOB_ID -y
```

Managed jobs clean up their own resources after cancellation.

### Tearing Down Services

```bash
sky serve down SERVICE_NAME -y
```

### Tearing Down Everything

If the user chose "all":

```bash
# Cancel all managed jobs first
sky jobs cancel -a -y

# Tear down all services
sky serve down SERVICE_NAME -y  # for each service

# Tear down all clusters
sky down -a -y
```

Execute `sky down -a` last because managed jobs and services may have associated clusters.

## Step 6: Verify and Report

After teardown, verify everything was cleaned up:

```bash
sky status
```

```bash
sky jobs queue
```

```bash
sky serve status
```

Present a final report:

```
=== TEARDOWN COMPLETE ===

Torn down:
  - Cluster 'train-01' (H100:8) -- REMOVED
  - Cluster 'dev' (A100:1) -- REMOVED

Remaining:
  - No active clusters
  - 1 managed job (ID 41, SUCCEEDED -- will auto-clean)
  - No active services

Cost savings: $9.60/hr ($230.40/day)
Total cost of torn-down resources: $55.20
```

### Handling Teardown Failures

If `sky down` fails (e.g., cloud provider error), report the error and suggest:

```
ERROR: Failed to tear down cluster 'train-01':
  Cloud API error: Instance not found

  This can happen if the instance was already terminated by the cloud provider.
  Try: sky down train-01 --purge
  This removes SkyPilot's local record without contacting the cloud.
```

If the failure is transient (network error), suggest retrying.

## Cleanup Reminder

After teardown, remind the user about other potential cost sources:
- **SkyPilot storage buckets**: `sky storage ls` to check for orphaned buckets
- **Cloud storage**: Checkpoint buckets in S3/GCS may still incur storage costs
- **Stopped clusters**: `sky status` may show STOPPED clusters that still cost disk money

## Reference

For CLI command details, see the skypilot-core skill at `/home/mikeb/skymcp/skills/skypilot-core/SKILL.md`.
