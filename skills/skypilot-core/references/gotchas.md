# SkyPilot Gotchas and Workarounds

21 known issues with solutions, ordered by frequency of impact.

## 1. Ray Port Conflict

**Problem**: SkyPilot uses Ray internally on port 6380. If your training code starts its own Ray cluster on 6380, both collide.

**Workaround**: Use port 6379 for your Ray cluster:
```python
ray.init(address="auto", _node_ip_address="0.0.0.0", dashboard_port=6379)
```

Or set in environment:
```yaml
envs:
  RAY_PORT: 6379
```

## 2. Post-v0.8.1 Async SDK Breaking Change

**Problem**: Before v0.8.1, `sky.launch()` returned results directly. After v0.8.1, all SDK calls are async and return `RequestId`.

**Before (v0.8.0 and earlier)**:
```python
job_id, handle = sky.launch(task, cluster_name="my-cluster")
```

**After (v0.8.1+)**:
```python
req = sky.launch(task, cluster_name="my-cluster")
job_id, handle = sky.get(req)
```

**Detection**: `TypeError: cannot unpack non-iterable RequestId object`

## 3. API Server Restart After Upgrade

**Problem**: After upgrading SkyPilot, the background API server may be running the old version.

**Fix**: Always run after upgrade:
```bash
sky api stop
# Next sky command will auto-start the new server
sky status
```

## 4. file_mounts Processed Before setup

**Problem**: `file_mounts` are mounted/copied before `setup` commands run. If you install a tool in `setup` that you need to process mounted files, the tool is not available during mounting.

**Workaround**: Do post-processing of mounted files in the `run` section, not `setup`.

## 5. MOUNT Mode Limitations

**Problem**: MOUNT mode is read-only via FUSE. No random writes, no appends, no new file creation.

**Symptoms**: `OSError: Read-only file system` or silent data corruption.

**Fix**: Use `MOUNT_CACHED` for any read-write workload:
```yaml
file_mounts:
  /data:
    source: s3://bucket/data
    mode: MOUNT_CACHED  # Not MOUNT
```

## 6. MOUNT_CACHED Disk Exhaustion

**Problem**: MOUNT_CACHED writes to local disk first, then syncs to cloud. If your write speed exceeds the upload bandwidth, local disk fills up.

**Symptoms**: `OSError: No space left on device` despite cloud bucket having space.

**Workaround**: Increase disk size and use high-tier disk:
```yaml
resources:
  disk_size: 2048
  disk_tier: high
```

Or add periodic sync waits in your training script.

## 7. Exposed Ports Are PUBLIC

**Problem**: `ports: [8080]` opens that port to the entire internet with zero authentication. Anyone can access your model endpoint, Jupyter notebook, or TensorBoard.

**Fix**: Add authentication middleware:
```python
# For FastAPI
from fastapi.security import HTTPBearer
security = HTTPBearer()

@app.get("/predict")
async def predict(token: str = Depends(security)):
    verify_token(token)
    ...
```

Or use SSH tunneling instead of public ports:
```bash
ssh -L 8080:localhost:8080 mycluster
```

## 8. .gitignore Respected for rsync

**Problem**: SkyPilot uses rsync to copy your workdir to the cluster, and it respects `.gitignore`. Files listed in `.gitignore` are NOT copied.

**Fix**: Create `.skyignore` to override. If `.skyignore` exists, it is used INSTEAD of `.gitignore`:
```
# .skyignore - only these patterns are excluded
__pycache__/
*.pyc
.env
```

## 9. Large workdir Files Are Slow

**Problem**: Everything in your working directory is rsynced to the cluster on every `sky launch` and `sky exec`. Large files (datasets, checkpoints, model weights) make this painfully slow.

**Fix**: Use `file_mounts` for anything over 1 GB:
```yaml
file_mounts:
  /models:
    source: s3://bucket/models
    mode: COPY
workdir: .  # Only code, configs, small files
```

## 10. sky storage ls Only Shows SkyPilot Buckets

**Problem**: `sky storage ls` only lists storage buckets created by SkyPilot. Pre-existing buckets referenced in `file_mounts` are not shown.

**Workaround**: Use cloud-native tools to list all storage:
```bash
aws s3 ls
gsutil ls
az storage container list
```

## 11. Kubernetes: Only Current Context Used

**Problem**: SkyPilot uses only the current kubectl context. If you have multiple clusters, you must switch context before running sky commands.

**Fix**: Set context before SkyPilot:
```bash
kubectl config use-context my-gpu-cluster
sky check
sky launch train.yaml
```

## 12. cost-report Only Tracks SkyPilot Lifecycle

**Problem**: `sky cost-report` only tracks costs for clusters managed by SkyPilot. Manually created instances, reserved instances, or non-SkyPilot resources are not tracked.

**Workaround**: Use cloud billing dashboards for total cost visibility. Tag SkyPilot resources:
```yaml
resources:
  labels:
    project: my-training
    team: ml
```

## 13. SSH to Worker Nodes Only on Interactive Clusters

**Problem**: `ssh mycluster` connects to the head node. You cannot SSH directly to worker nodes in managed jobs (`sky jobs launch`).

**Workaround**: Use head node as jump host for interactive clusters:
```bash
# Get worker IPs
sky exec mycluster 'echo $SKYPILOT_NODE_IPS'
# SSH to worker via head
ssh -J mycluster worker-ip
```

For managed jobs, use `sky jobs logs` instead.

## 14. .git Directory Excluded in Managed Jobs

**Problem**: Managed jobs strip the `.git` directory during upload. Git commands will fail inside managed jobs.

**Workaround**: If you need git in managed jobs, clone the repo in `setup`:
```yaml
setup: |
  git clone https://github.com/myorg/myrepo.git /code
run: |
  cd /code && python train.py
```

## 15. AWS SSO Only Works with MOUNT_CACHED

**Problem**: AWS SSO (IAM Identity Center) credentials may not work with MOUNT mode (FUSE mounts use long-lived credentials). MOUNT_CACHED works because it uses the AWS SDK with automatic credential refresh.

**Fix**: Use MOUNT_CACHED when using AWS SSO:
```yaml
file_mounts:
  /data:
    source: s3://bucket/data
    mode: MOUNT_CACHED  # Works with SSO
```

## 16. Symbolic Links Not Copied in file_mounts

**Problem**: Symlinks in your workdir or file_mounts source are not followed. The symlink itself may be copied, but it will be broken on the cluster.

**Fix**: Resolve symlinks before copying, or use real files:
```bash
# Before sky launch, resolve symlinks
cp -rL symlinked_dir/ real_dir/
```

## 17. File Permissions Not Preserved in Bucket Attachment

**Problem**: Files uploaded to cloud storage lose their Unix permissions. Executable scripts become non-executable.

**Fix**: Add `chmod` in setup or run:
```yaml
run: |
  chmod +x /data/scripts/*.sh
  /data/scripts/train.sh
```

## 18. Cluster Name Collisions

**Problem**: If two users use the same cluster name in the same cloud account, they may overwrite each other's clusters.

**Fix**: Use unique prefixes:
```bash
sky launch train.yaml -c ${USER}-training-$(date +%Y%m%d)
```

## 19. Setup Section Caching

**Problem**: SkyPilot caches the `setup` section by hash. If you change a dependency version but the setup commands look the same (e.g., `pip install -r requirements.txt` where requirements.txt changed), the cached setup may be used.

**Fix**: Force re-run by changing the setup text:
```yaml
setup: |
  # cache-bust: v2
  pip install -r requirements.txt
```

Or use `sky launch --no-setup` followed by `sky exec` with setup.

## 20. Multi-Node HEAD_IP Race Condition

**Problem**: In multi-node setups, `SKYPILOT_NODE_IPS` is populated before all nodes are ready. If the master starts training before workers join, it may fail.

**Fix**: Add a barrier in your run script:
```bash
run: |
  # Wait for all nodes to be reachable
  for ip in $(echo "$SKYPILOT_NODE_IPS" | tail -n +2); do
    until ssh -o ConnectTimeout=5 $ip true 2>/dev/null; do
      echo "Waiting for $ip..."
      sleep 5
    done
  done
  echo "All nodes ready"
  torchrun ...
```

## 21. Cloud Quota Limits

**Problem**: `sky launch` fails with quota errors when you have not requested GPU quota increases from your cloud provider.

**Symptoms**: `QUOTA_EXCEEDED`, `InsufficientInstanceCapacity`, or generic provision failures.

**Fix**: Request quota increases ahead of time:
- **AWS**: Service Quotas console -> EC2 -> Running On-Demand/Spot instances
- **GCP**: IAM & Admin -> Quotas -> GPUs (all regions)
- **Azure**: Subscription -> Usage + quotas -> Compute
- Use `sky check` to verify credentials and `sky gpus list` to check pricing/availability
