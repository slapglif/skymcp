# SkyPilot YAML Specification

Complete reference for all fields in a SkyPilot task YAML.

## Top-Level Fields

```yaml
# Task name (used in logs, status, cost reports)
name: my-training-run

# Resource requirements
resources:
  cloud: aws                    # aws, gcp, azure, lambda, runpod, fluidstack, cudo, ibm, oci, scp, vsphere, kubernetes
  region: us-east-1             # Specific region (optional)
  zone: us-east-1a             # Specific zone (optional)
  accelerators: A100:4          # GPU_TYPE:COUNT or CPU:N
  cpus: 32+                     # Minimum CPUs (N or N+)
  memory: 128+                  # Minimum memory in GB (N or N+)
  instance_type: p4d.24xlarge   # Specific instance (overrides accelerators)
  use_spot: true                # Spot/preemptible instances
  spot_recovery: FAILOVER       # Deprecated: use job_recovery
  disk_size: 512                # OS disk in GB (default: 256)
  disk_tier: high               # low, medium, high, best, ultra
  network_tier: premium         # standard, premium (GCP only)
  ports:                        # Expose ports publicly
    - 8080
    - 8888
  accelerator_args:             # GPU-specific args
    runtime_version: tpu-vm-base  # TPU runtime
  labels:                       # Cloud-specific labels
    team: ml-research

  # Multi-cloud failover
  any_of:                       # Try any in parallel
    - accelerators: A100:8
      cloud: gcp
    - accelerators: A100:8
      cloud: aws

  ordered:                      # Try sequentially (priority)
    - accelerators: H100:8
      use_spot: true
    - accelerators: A100:8
      use_spot: true
    - accelerators: A100:8
      use_spot: false

  # Managed job recovery
  job_recovery:
    strategy: FAILOVER           # FAILOVER (try other regions/clouds) or NONE
    max_restarts_on_errors: 3    # Max restarts on application errors (non-preemption)

# Multi-node cluster
num_nodes: 4                     # Number of nodes

# Environment variables
envs:
  WANDB_API_KEY: null            # null = read from local environment
  HF_TOKEN: null
  BATCH_SIZE: 32                 # Literal value
  MY_VAR: ${HOME}/data           # Shell expansion happens at runtime

# Secrets (files mounted securely)
secrets:
  - name: my-api-key
    source: ~/.secrets/api-key.txt    # Local file
    target: /run/secrets/api-key.txt  # Mount path on cluster

# File mounts
file_mounts:
  # Cloud storage mount
  /data:
    source: s3://my-bucket/datasets
    mode: MOUNT                  # MOUNT, COPY, or MOUNT_CACHED

  # Local files uploaded via rsync
  /code/config.yaml:
    source: ./config.yaml        # Relative to YAML location

  # Cloud storage bucket (auto-created if needed)
  /checkpoints:
    name: my-checkpoint-bucket   # Bucket name
    source: s3://my-checkpoint-bucket
    mode: MOUNT_CACHED
    store: s3                    # s3, gcs, azure, r2, ibm

# Working directory (rsynced to cluster)
workdir: .                       # Current directory (default)

# Setup commands (run once at provision, cached after)
setup: |
  pip install torch transformers wandb
  git clone https://github.com/myorg/myrepo.git

# Run commands (run every time on launch/exec)
run: |
  cd /code
  torchrun --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE train.py
```

## Storage Modes Detail

### MOUNT

```yaml
file_mounts:
  /data:
    source: s3://bucket/path
    mode: MOUNT
```

- Streams data from cloud storage on access (FUSE mount)
- **Read-only**: No random writes, no appends, no creates
- Zero provision time (no download)
- Good for: Large read-only datasets, streaming access
- Bad for: Checkpoints, any writes, random access patterns

### COPY

```yaml
file_mounts:
  /data:
    source: s3://bucket/path
    mode: COPY
```

- Downloads entire contents at provision time
- Full read-write access on local disk
- Provision time proportional to data size
- Changes are local only (not synced back)
- Good for: Small datasets, code, configs
- Bad for: Multi-TB datasets (slow provision)

### MOUNT_CACHED

```yaml
file_mounts:
  /checkpoints:
    source: s3://bucket/checkpoints
    mode: MOUNT_CACHED
```

- Local VFS cache with background cloud sync
- Read and write support
- Reads: served from local cache after first access
- Writes: written locally, synced to cloud asynchronously
- Good for: Checkpoints, model weights, iterative read-write
- Bad for: Nothing (most versatile mode)
- **Warning**: Local disk can fill if write speed exceeds upload speed

## Multi-Stage Pipelines

Separate stages with `---`. Each stage can have different resources:

```yaml
name: preprocess
resources:
  cpus: 32
run: python preprocess.py

---

name: train
resources:
  accelerators: A100:8
  disk_size: 1024
num_nodes: 4
run: |
  torchrun --nnodes=$SKYPILOT_NUM_NODES \
    --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
    --node_rank=$SKYPILOT_NODE_RANK \
    --master_addr=$(echo "$SKYPILOT_NODE_IPS" | head -n1) \
    --master_port=12345 \
    train.py

---

name: evaluate
resources:
  accelerators: A100:1
run: python eval.py --checkpoint /checkpoints/best
```

## SkyServe YAML

Service-specific fields for `sky serve`:

```yaml
service:
  readiness_probe:
    path: /health                # Health check endpoint
    post_data: null              # POST body (null = GET)
    initial_delay_seconds: 120   # Wait before first probe
    timeout_seconds: 10          # Probe timeout

  replicas: 2                    # Fixed replica count
  # OR auto-scaling:
  replica_policy:
    min_replicas: 1
    max_replicas: 8
    target_qps_per_replica: 10   # Scale based on QPS
    upscale_delay_seconds: 60
    downscale_delay_seconds: 300

  # Load balancing
  load_balancer:
    policy: round_robin          # round_robin or least_connections

resources:
  accelerators: A100:1
  ports:
    - 8080

run: |
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8b \
    --port 8080
```

## Controller YAML

For managed jobs with spot recovery:

```yaml
name: long-training

resources:
  accelerators: A100:8
  use_spot: true
  job_recovery:
    strategy: FAILOVER
    max_restarts_on_errors: 3

file_mounts:
  /checkpoints:
    source: s3://my-bucket/checkpoints
    mode: MOUNT_CACHED

run: |
  # Resume from checkpoint if exists
  CKPT=$(find /checkpoints -name "step_*" | sort -V | tail -1)
  if [ -n "$CKPT" ]; then
    echo "Resuming from $CKPT"
    python train.py --resume $CKPT
  else
    echo "Starting fresh"
    python train.py
  fi
```

## Common YAML Patterns

### GPU Development Cluster with Auto-Stop

```yaml
name: dev-cluster
resources:
  accelerators: A100:1
  ports:
    - 8888  # Jupyter
    - 6006  # TensorBoard
setup: |
  pip install jupyterlab tensorboard
run: |
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser &
  sleep infinity
```

```bash
sky launch dev.yaml -c dev --idle-minutes-to-autostop 60
```

### Multi-Cloud Training with Checkpointing

```yaml
name: resilient-training
resources:
  ordered:
    - accelerators: H100:8
      cloud: lambda
      use_spot: true
    - accelerators: A100:8
      cloud: gcp
      use_spot: true
    - accelerators: A100:8
      cloud: aws
      use_spot: false
  job_recovery:
    strategy: FAILOVER
    max_restarts_on_errors: 5

file_mounts:
  /data:
    source: s3://training-data/dataset
    mode: MOUNT
  /checkpoints:
    source: s3://training-data/checkpoints
    mode: MOUNT_CACHED

envs:
  WANDB_API_KEY: null
  WANDB_PROJECT: my-training

setup: |
  pip install torch transformers wandb

run: |
  python train.py \
    --data /data \
    --checkpoint-dir /checkpoints \
    --auto-resume
```

### Hyperparameter Sweep

```yaml
name: sweep-${HP_LR}-${HP_BS}
resources:
  accelerators: A100:1
  use_spot: true

envs:
  HP_LR: 1e-4
  HP_BS: 32

run: |
  python train.py --lr $HP_LR --batch-size $HP_BS
```

Launch multiple:
```bash
for lr in 1e-3 1e-4 1e-5; do
  for bs in 16 32 64; do
    sky jobs launch sweep.yaml --env HP_LR=$lr --env HP_BS=$bs
  done
done
```
