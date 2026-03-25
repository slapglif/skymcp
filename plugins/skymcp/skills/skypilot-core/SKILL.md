---
name: skypilot-core
description: Use when working with SkyPilot CLI commands, sky launch, sky exec, sky jobs, managed jobs, sky serve, sky status, sky yaml configuration, cloud GPU provisioning, sky check credentials, or the SkyPilot Python SDK - the authoritative SkyPilot reference for all cloud ML infrastructure operations
---

# SkyPilot Core Reference

## CLI Command Reference

| Command | Purpose | Example |
|---------|---------|---------|
| `sky launch [YAML] -c NAME` | Provision cluster and run task | `sky launch train.yaml -c mycluster` |
| `sky exec NAME [YAML]` | Run task on existing cluster (skip setup) | `sky exec mycluster eval.yaml` |
| `sky stop NAME` | Stop cluster (keep disk, stop billing compute) | `sky stop mycluster` |
| `sky start NAME` | Restart stopped cluster | `sky start mycluster` |
| `sky down NAME` | Tear down cluster completely | `sky down mycluster` |
| `sky status [-r]` | List clusters (`-r` for all regions) | `sky status` |
| `sky jobs launch YAML` | Submit managed job (auto-recovery) | `sky jobs launch train.yaml` |
| `sky jobs queue` | List managed jobs | `sky jobs queue` |
| `sky jobs logs JOB_ID` | Stream managed job logs | `sky jobs logs 42` |
| `sky jobs cancel JOB_ID` | Cancel managed job | `sky jobs cancel 42` |
| `sky serve up YAML` | Deploy SkyServe endpoint | `sky serve up serve.yaml` |
| `sky serve down NAME` | Tear down service | `sky serve down myservice` |
| `sky serve status` | List services and replicas | `sky serve status` |
| `sky serve update NAME YAML` | Rolling update a service | `sky serve update myservice serve-v2.yaml` |
| `sky gpus list [GPU:N]` | Show GPU pricing across clouds | `sky gpus list A100:8` |
| `sky cost-report` | Show spending on SkyPilot-managed clusters | `sky cost-report` |
| `sky check` | Verify cloud credentials | `sky check` |
| `sky storage ls` | List SkyPilot-created cloud storage | `sky storage ls` |
| `sky storage delete NAME` | Delete SkyPilot-created storage | `sky storage delete my-bucket` |

## Key `sky launch` Flags

| Flag | Purpose | Default |
|------|---------|---------|
| `-c NAME` | Cluster name (reuse existing) | auto-generated |
| `--gpus GPU[:N]` | Request GPU type and count | from YAML |
| `--use-spot` | Request spot/preemptible instances | false |
| `--retry-until-up` | Retry on capacity errors | false |
| `--idle-minutes-to-autostop N` | Auto-stop after N idle minutes | disabled |
| `--down` | Tear down after task completes | false |
| `--env KEY=VAL` | Set environment variable | none |
| `--ports N` | Expose port to public internet | none |
| `--async` | Return immediately, do not stream logs | false |
| `--dryrun` | Show what would happen without executing | false |
| `--num-nodes N` | Multi-node cluster | 1 |
| `--disk-size N` | OS disk size in GB | 256 |
| `--disk-tier {low,medium,high,best,ultra}` | Disk performance tier | medium |
| `-y` | Skip confirmation prompt | false |

## YAML Spec Essentials

Minimal training YAML:

```yaml
name: my-training-run

resources:
  accelerators: A100:4
  use_spot: true
  disk_size: 512
  disk_tier: high
  ports:
    - 8080

num_nodes: 1

envs:
  WANDB_API_KEY: null  # null = read from local env

file_mounts:
  /data:
    source: s3://my-bucket/datasets
    mode: MOUNT_CACHED

setup: |
  pip install torch transformers wandb

run: |
  torchrun --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE train.py
```

### Resource Selection

Use `any_of` for multi-cloud failover:

```yaml
resources:
  any_of:
    - accelerators: A100:8
      cloud: gcp
    - accelerators: A100:8
      cloud: aws
    - accelerators: H100:8
      cloud: lambda
```

Use `ordered` for priority failover (try first option, fall back to next):

```yaml
resources:
  ordered:
    - accelerators: H100:8
      use_spot: true
    - accelerators: A100:8
      use_spot: true
    - accelerators: A100:8
      use_spot: false
```

### Storage Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `MOUNT` | Stream from bucket on access | Read-only datasets, streaming |
| `COPY` | Download entire contents at provision time | Small datasets, code |
| `MOUNT_CACHED` | Local VFS cache + bucket sync | Checkpoints, write-heavy workloads |

### Multi-Stage Pipelines

Separate stages with `---` in a single YAML. Each stage runs sequentially:

```yaml
name: preprocess
run: python preprocess.py
---
name: train
resources:
  accelerators: A100:4
run: torchrun train.py
---
name: eval
run: python eval.py
```

### Job Recovery for Spot Instances

```yaml
resources:
  use_spot: true
  job_recovery:
    strategy: FAILOVER
    max_restarts_on_errors: 3
```

## Environment Variables

SkyPilot injects these into every task:

| Variable | Value | Notes |
|----------|-------|-------|
| `SKYPILOT_NODE_RANK` | 0, 1, 2, ... | Rank of this node in multi-node |
| `SKYPILOT_NODE_IPS` | Newline-separated IPs | All node internal IPs |
| `SKYPILOT_NUM_NODES` | Total node count | For distributed launch |
| `SKYPILOT_NUM_GPUS_PER_NODE` | GPUs on this node | For torchrun nproc |
| `SKYPILOT_TASK_ID` | Stable UUID | Survives spot preemption |
| `SKYPILOT_CLUSTER_INFO` | JSON blob | Cloud, region, instance type |

## Python SDK (Post-v0.8.1)

All SDK calls are async and return a `RequestId`. Retrieve results with `sky.get()`:

```python
import sky

task = sky.Task.from_yaml("train.yaml")

# Launch returns RequestId, not result directly
req = sky.launch(task, cluster_name="my-cluster")
job_id, handle = sky.get(req)

# Same pattern for all operations
req = sky.status()
clusters = sky.get(req)

req = sky.exec(task, cluster_name="my-cluster")
job_id, handle = sky.get(req)

req = sky.down("my-cluster")
sky.get(req)
```

### Task Construction Programmatically

```python
task = sky.Task(
    name="my-training",
    setup="pip install torch",
    run="python train.py",
)
task.set_resources(sky.Resources(
    accelerators="A100:4",
    use_spot=True,
    disk_size=512,
))
task.set_file_mounts({
    "/data": sky.Storage(source="s3://bucket/data", mode=sky.StorageMode.MOUNT_CACHED),
})
```

## Managed Jobs vs Interactive Clusters

| Feature | Interactive (`sky launch`) | Managed (`sky jobs launch`) |
|---------|--------------------------|---------------------------|
| SSH access | Yes | No |
| Survives preemption | No (must restart) | Yes (auto-recovery) |
| Cost | Pay while idle | Pay only during execution |
| Logs | `sky logs CLUSTER` | `sky jobs logs JOB_ID` |
| Lifecycle | Manual stop/down | Auto-cleanup on completion |
| Use case | Development, debugging | Production training runs |

## SkyServe Quick Reference

```yaml
# serve.yaml
service:
  readiness_probe:
    path: /health
    initial_delay_seconds: 120
  replicas: 2

resources:
  accelerators: A100:1

run: |
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8b \
    --port 8080
```

```bash
sky serve up serve.yaml -n my-llm
# Returns endpoint URL: http://44.123.456.78:30001
sky serve status          # Check replicas
sky serve update my-llm serve-v2.yaml  # Rolling update
sky serve down my-llm     # Tear down
```

## Common Workflows

### Develop Interactively, Then Submit as Managed Job

```bash
# 1. Develop on interactive cluster
sky launch -c dev --gpus A100:1 --idle-minutes-to-autostop 30
ssh dev
# ... iterate on code ...

# 2. Submit as managed job for production run
sky jobs launch train.yaml
sky jobs queue  # Monitor
sky jobs logs 42  # Stream logs
```

### Multi-Node Distributed Training

```yaml
num_nodes: 4
resources:
  accelerators: H100:8

run: |
  torchrun \
    --nnodes=$SKYPILOT_NUM_NODES \
    --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
    --node_rank=$SKYPILOT_NODE_RANK \
    --master_addr=$(echo "$SKYPILOT_NODE_IPS" | head -n1) \
    --master_port=12345 \
    train.py
```

### Secrets Management

```yaml
envs:
  WANDB_API_KEY: null   # Reads from local environment
  HF_TOKEN: null        # Reads from local environment

secrets:
  - name: my-secret
    source: ~/.secrets/api-key.txt
    target: /run/secrets/api-key.txt
```

## Top Gotchas

Avoid these common pitfalls. See [references/gotchas.md](references/gotchas.md) for the full list with workarounds.

1. **Ray port conflict** -- SkyPilot uses port 6380; use 6379 for your own Ray cluster
2. **Post-v0.8.1 async SDK** -- All Python SDK calls return `RequestId`; wrap with `sky.get()`
3. **Restart API server after upgrade** -- Run `sky api stop` then retry after upgrading SkyPilot
4. **file_mounts run before setup** -- Mount sources must exist before setup commands run
5. **MOUNT mode is read-only** -- No random writes, no appends; use MOUNT_CACHED for writes
6. **Exposed ports are PUBLIC** -- Add authentication middleware; SkyPilot does not gate access
7. **Large workdir is slow** -- rsync copies entire workdir; use file_mounts for files over 1 GB

Full YAML spec: [references/yaml-spec.md](references/yaml-spec.md)
Full Python SDK: [references/python-sdk.md](references/python-sdk.md)
Full gotchas list: [references/gotchas.md](references/gotchas.md)
