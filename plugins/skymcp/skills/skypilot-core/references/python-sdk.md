# SkyPilot Python SDK Reference

Complete reference for the SkyPilot Python SDK (post-v0.8.1).

## Critical: Async API Pattern

All SDK calls in v0.8.1+ are asynchronous. Every function returns a `RequestId`. Use `sky.get()` to retrieve results:

```python
import sky

# All calls return RequestId
req = sky.launch(task, cluster_name="my-cluster")
result = sky.get(req)  # Blocks until complete

# DO NOT do this (pre-v0.8.1 pattern, will fail):
# job_id, handle = sky.launch(task, cluster_name="my-cluster")  # TypeError
```

## Task Construction

### From YAML

```python
task = sky.Task.from_yaml("train.yaml")
```

### Programmatic

```python
task = sky.Task(
    name="my-training",
    setup="pip install torch transformers",
    run="python train.py --lr 1e-4",
    envs={"WANDB_API_KEY": None, "BATCH_SIZE": "32"},
    workdir=".",
    num_nodes=1,
)
```

### Resources

```python
resources = sky.Resources(
    cloud=sky.AWS(),               # or sky.GCP(), sky.Azure(), sky.Lambda(), etc.
    region="us-east-1",
    accelerators="A100:4",
    cpus="32+",
    memory="128+",
    use_spot=True,
    disk_size=512,
    disk_tier="high",
    ports=[8080, 8888],
)
task.set_resources(resources)
```

### Multi-Cloud Failover

```python
resources_options = [
    sky.Resources(cloud=sky.GCP(), accelerators="A100:8"),
    sky.Resources(cloud=sky.AWS(), accelerators="A100:8"),
    sky.Resources(cloud=sky.Lambda(), accelerators="H100:8"),
]
task.set_resources(resources_options)  # any_of semantics
```

### File Mounts

```python
task.set_file_mounts({
    "/data": sky.Storage(
        source="s3://my-bucket/datasets",
        mode=sky.StorageMode.MOUNT_CACHED,
    ),
    "/code/config.yaml": "./config.yaml",  # Local file
})
```

### Storage Object

```python
storage = sky.Storage(
    name="my-bucket",
    source="s3://my-bucket/data",
    mode=sky.StorageMode.MOUNT_CACHED,  # MOUNT, COPY, MOUNT_CACHED
    store=sky.StoreType.S3,             # S3, GCS, AZURE, R2, IBM
)
```

## Cluster Lifecycle

### Launch (Provision + Run)

```python
req = sky.launch(
    task,
    cluster_name="my-cluster",
    retry_until_up=True,          # Retry on capacity errors
    idle_minutes_to_autostop=60,  # Auto-stop after 60 idle minutes
    down=False,                   # If True, tear down after task
    dryrun=False,                 # If True, show plan only
)
job_id, handle = sky.get(req)
print(f"Job {job_id} launched on {handle}")
```

### Execute on Existing Cluster

```python
req = sky.exec(
    task,
    cluster_name="my-cluster",
    down=False,
    dryrun=False,
)
job_id, handle = sky.get(req)
```

### Stop Cluster

```python
req = sky.stop("my-cluster")
sky.get(req)
```

### Start Stopped Cluster

```python
req = sky.start("my-cluster")
sky.get(req)
```

### Tear Down Cluster

```python
req = sky.down("my-cluster")
sky.get(req)
```

### Autostop Configuration

```python
req = sky.autostop(
    "my-cluster",
    idle_minutes=30,
    down=True,  # Tear down instead of stop
)
sky.get(req)
```

## Status and Monitoring

### List Clusters

```python
req = sky.status()
clusters = sky.get(req)
for cluster in clusters:
    print(f"{cluster['name']}: {cluster['status']}")
```

### Get Cluster Details

```python
req = sky.status(cluster_names=["my-cluster"])
details = sky.get(req)
```

### Cost Report

```python
req = sky.cost_report()
report = sky.get(req)
for entry in report:
    print(f"{entry['name']}: ${entry['total_cost']:.2f}")
```

## Managed Jobs

### Launch Managed Job

```python
req = sky.jobs.launch(
    task,
    name="my-training-job",
)
job_id = sky.get(req)
```

### List Jobs

```python
req = sky.jobs.queue()
jobs = sky.get(req)
for job in jobs:
    print(f"Job {job['id']}: {job['status']}")
```

### Cancel Job

```python
req = sky.jobs.cancel(job_id=42)
sky.get(req)
```

## SkyServe

### Deploy Service

```python
req = sky.serve.up(task, service_name="my-llm")
sky.get(req)
```

### Get Service Status

```python
req = sky.serve.status()
services = sky.get(req)
```

### Update Service (Rolling)

```python
new_task = sky.Task.from_yaml("serve-v2.yaml")
req = sky.serve.update("my-llm", new_task)
sky.get(req)
```

### Tear Down Service

```python
req = sky.serve.down("my-llm")
sky.get(req)
```

## Utility Functions

### Check Credentials

```python
sky.check.check()
# Prints enabled clouds and credential status
```

### List Available GPUs

```python
req = sky.status(gpus=True)
gpus = sky.get(req)
```

## Complete Example: Training Pipeline

```python
import sky
import time

# Define preprocessing task
preprocess_task = sky.Task(
    name="preprocess",
    setup="pip install datasets transformers",
    run="python preprocess.py --output /data/processed",
)
preprocess_task.set_resources(sky.Resources(cpus="32+", memory="64+"))
preprocess_task.set_file_mounts({
    "/data": sky.Storage(source="s3://my-bucket/data", mode=sky.StorageMode.MOUNT_CACHED),
})

# Define training task
train_task = sky.Task(
    name="train",
    setup="pip install torch transformers wandb",
    run="torchrun --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE train.py",
    envs={"WANDB_API_KEY": None},
    num_nodes=2,
)
train_task.set_resources(sky.Resources(
    accelerators="A100:8",
    use_spot=True,
    disk_size=1024,
    disk_tier="high",
))
train_task.set_file_mounts({
    "/data": sky.Storage(source="s3://my-bucket/data", mode=sky.StorageMode.MOUNT),
    "/checkpoints": sky.Storage(source="s3://my-bucket/ckpts", mode=sky.StorageMode.MOUNT_CACHED),
})

# Launch preprocessing
print("Starting preprocessing...")
req = sky.launch(preprocess_task, cluster_name="preprocess-cluster", down=True)
sky.get(req)
print("Preprocessing complete")

# Launch training as managed job (auto-recovery)
print("Starting training...")
req = sky.jobs.launch(train_task, name="training-run")
job_id = sky.get(req)
print(f"Training job {job_id} submitted")

# Monitor
while True:
    req = sky.jobs.queue()
    jobs = sky.get(req)
    job = next((j for j in jobs if j["id"] == job_id), None)
    if job and job["status"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
        print(f"Job {job_id} finished: {job['status']}")
        break
    time.sleep(60)
```

## Error Handling

```python
import sky
from sky.exceptions import (
    ResourcesUnavailableError,
    ClusterNotUpError,
    CommandError,
)

try:
    req = sky.launch(task, cluster_name="my-cluster", retry_until_up=True)
    sky.get(req)
except ResourcesUnavailableError:
    print("No capacity available. Try different GPU or region.")
except ClusterNotUpError:
    print("Cluster failed to start. Check credentials with sky check.")
except CommandError as e:
    print(f"Command failed: {e}")
```

## Environment Detection

Inside a SkyPilot task, detect the runtime environment:

```python
import os
import json

# Am I running on SkyPilot?
is_skypilot = "SKYPILOT_TASK_ID" in os.environ

# My node rank in multi-node
node_rank = int(os.environ.get("SKYPILOT_NODE_RANK", 0))

# All node IPs
node_ips = os.environ.get("SKYPILOT_NODE_IPS", "").strip().split("\n")

# Number of GPUs on this node
num_gpus = int(os.environ.get("SKYPILOT_NUM_GPUS_PER_NODE", 1))

# Cluster info
cluster_info = json.loads(os.environ.get("SKYPILOT_CLUSTER_INFO", "{}"))
cloud = cluster_info.get("cloud", "unknown")
region = cluster_info.get("region", "unknown")
```
