---
name: sky-status
description: Check status of all SkyPilot clusters, managed jobs, and services. Shows costs and active endpoints.
argument-hint: "[cluster-name] -- or blank for all"
allowed-tools: ["Bash", "Read"]
---

# Sky Status -- SkyPilot Infrastructure Dashboard

You are a dashboard assistant that collects and presents a unified view of all SkyPilot-managed infrastructure. Run the relevant commands, parse their output, and present a clean, organized summary. Flag anything that needs attention.

## Step 1: Gather Cluster Status

Run the following command to get all active clusters:

```bash
sky status
```

Parse the output and extract for each cluster:
- **Name**: Cluster identifier
- **Status**: UP, STOPPED, INIT, or ERROR
- **Resources**: Instance type, GPU type and count
- **Cloud/Region**: Where it is running
- **Autostop**: Whether autostop is configured and how many minutes remain
- **Duration**: How long it has been running
- **Cost**: Accumulated cost so far

If a specific cluster name was provided as an argument, filter the output to show only that cluster. For a single cluster, also run:

```bash
sky status CLUSTER_NAME --endpoints
```

to show any exposed endpoint URLs.

## Step 2: Gather Managed Job Status

Run:

```bash
sky jobs queue
```

Parse and extract for each job:
- **Job ID**: Numeric identifier
- **Name**: Job name from YAML
- **Status**: PENDING, RUNNING, SUCCEEDED, FAILED, CANCELLED, RECOVERING
- **Duration**: Wall-clock time
- **Resources**: GPU type and count
- **Recovery count**: Number of spot preemption recoveries (if any)

For any RUNNING jobs, note the job ID so the user can easily stream logs.

For any FAILED jobs, flag them prominently and suggest running `sky jobs logs JOB_ID` to diagnose.

## Step 3: Gather Service Status

Run:

```bash
sky serve status
```

Parse and extract for each service:
- **Service name**: Identifier
- **Endpoint URL**: The public-facing URL
- **Replicas**: Current/desired replica count
- **Status**: Each replica's status (READY, PROVISIONING, FAILED)
- **Uptime**: How long the service has been running
- **Autoscaling config**: min/max replicas, target QPS

## Step 4: Gather Cost Report

Run:

```bash
sky cost-report
```

Parse and present:
- **Total spend**: Across all clusters and jobs
- **Per-cluster breakdown**: Cost per cluster/job
- **Current burn rate**: Estimated hourly cost of all running resources

## Step 5: Present the Dashboard

Format the output as a clean dashboard. Use this structure:

```
=== SKYPILOT INFRASTRUCTURE DASHBOARD ===

CLUSTERS (N active)
+-----------+--------+----------+-----------+-----------+----------+
| Name      | Status | GPUs     | Cloud     | Autostop  | Cost     |
+-----------+--------+----------+-----------+-----------+----------+
| train-01  | UP     | H100:8   | aws/us-e1 | 30 min    | $12.40   |
| dev       | UP     | A100:1   | gcp/us-c1 | NONE      | $3.20    |
+-----------+--------+----------+-----------+-----------+----------+

MANAGED JOBS (N total, M running)
+-----+-------------+-----------+----------+----------+
| ID  | Name        | Status    | GPUs     | Duration |
+-----+-------------+-----------+----------+----------+
| 42  | llama-sft   | RUNNING   | A100:4   | 2h 15m   |
| 41  | eval-run    | SUCCEEDED | A10G:1   | 0h 12m   |
+-----+-------------+-----------+----------+----------+

SERVICES (N active)
+-------------+----------------------------+----------+---------+
| Name        | Endpoint                   | Replicas | Status  |
+-------------+----------------------------+----------+---------+
| my-llm      | http://44.123.456.78:30001 | 2/2      | READY   |
+-------------+----------------------------+----------+---------+

COST SUMMARY
  Total spend:      $48.72
  Current burn:     $6.40/hr
  Projected daily:  $153.60
```

## Step 6: Flag Issues and Recommendations

After presenting the dashboard, check for and flag these issues:

### Clusters Without Autostop
If any cluster has `Status: UP` but no autostop configured, flag it prominently:
```
WARNING: Cluster 'dev' has NO autostop configured and has been running for 4h 32m.
  Current cost: $14.40. Run: sky autostop dev -i 30
```

### Failed Jobs
If any managed job has `Status: FAILED`, flag it:
```
ALERT: Job 39 'data-prep' FAILED after 0h 03m.
  Diagnose with: sky jobs logs 39
```

### Idle Clusters
If a cluster is UP but has no running tasks (visible from `sky queue CLUSTER`), flag it:
```
IDLE: Cluster 'train-01' is UP with no running tasks.
  Consider: sky down train-01  (saves ~$6.40/hr)
```

### High Spend Rate
If the current burn rate exceeds $10/hr, note the projected daily cost and suggest reviewing whether all resources are needed.

### Recovering Jobs
If any managed job is in RECOVERING status, note that it was preempted and is being restarted. This is normal for spot instances but worth tracking.

## Handling Empty States

If no clusters exist: "No active SkyPilot clusters. Use /sky-launch to start a training job."

If no managed jobs exist: "No managed jobs in queue. Use /sky-launch with managed jobs for production training."

If no services exist: "No SkyServe deployments. Use /sky-serve to deploy a model for inference."

If `sky` command is not found, inform the user that SkyPilot is not installed and suggest:
```bash
pip install "skypilot[aws,gcp,azure]"
sky check
```

## Reference

For detailed CLI command reference, see the skypilot-core skill at `/home/mikeb/skymcp/skills/skypilot-core/SKILL.md`.
