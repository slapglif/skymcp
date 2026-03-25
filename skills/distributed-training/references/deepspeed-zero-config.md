# DeepSpeed ZeRO Configuration Reference

Complete reference for all DeepSpeed ZeRO configuration parameters. Copy and adapt these configs for your training runs.

## Minimal Configs by Stage

### ZeRO-1 (Shard Optimizer Only)

Lowest overhead. Same communication as DDP. Good default starting point.

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 1
  },
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

### ZeRO-2 (Shard Optimizer + Gradients)

Best balance of memory savings and throughput for most workloads.

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  },
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

### ZeRO-3 (Shard Everything)

Maximum memory savings. Use when model does not fit with ZeRO-2.

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "none"
    },
    "offload_param": {
      "device": "none"
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

### ZeRO-3 + CPU Offload (Maximum Memory)

Last resort when GPU memory is insufficient. 2-5x slower than GPU-only.

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

### ZeRO-3 + NVMe Offload (ZeRO-Infinity)

Offload to NVMe SSD. Even slower than CPU offload but enables training models that exceed CPU memory.

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "nvme",
      "pin_memory": true,
      "nvme_path": "/local_nvme"
    },
    "offload_param": {
      "device": "nvme",
      "pin_memory": true,
      "nvme_path": "/local_nvme"
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto"
}
```

## Full Parameter Reference

### Top-Level Parameters

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0,
  "steps_per_print": 100,
  "wall_clock_breakdown": false,
  "dump_state": false
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `train_batch_size` | int/"auto" | Global batch size. Must = micro_batch * accum_steps * world_size |
| `train_micro_batch_size_per_gpu` | int/"auto" | Per-GPU batch size per forward pass |
| `gradient_accumulation_steps` | int/"auto" | Steps before optimizer update |
| `gradient_clipping` | float | Max gradient norm. 1.0 is standard. |
| `steps_per_print` | int | Log interval |
| `wall_clock_breakdown` | bool | Detailed timing breakdown |

**Note:** `"auto"` values are resolved by HuggingFace Trainer or your training script.

### Precision (bf16/fp16)

```json
{
  "bf16": {
    "enabled": true
  }
}
```

OR

```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  }
}
```

**Recommendation:** Use `bf16` on Ampere+ GPUs (A100, H100). Use `fp16` only on older GPUs (V100, T4).

### ZeRO Optimization Parameters

#### Common Parameters (All Stages)

```json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stage` | 0 | ZeRO stage (0=disabled, 1, 2, 3) |
| `allgather_partitions` | true | Use AllGather for parameter partitions |
| `allgather_bucket_size` | 5e8 | Bucket size for AllGather (bytes). Larger = more memory, fewer comms. |
| `overlap_comm` | false | Overlap communication with computation |
| `reduce_scatter` | true | Use ReduceScatter instead of AllReduce |
| `reduce_bucket_size` | 5e8 | Bucket size for gradient reduction (bytes) |
| `contiguous_gradients` | true | Copy gradients to contiguous buffer |

#### Stage 3 Specific Parameters

```json
{
  "zero_optimization": {
    "stage": 3,
    "sub_group_size": 1e9,
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sub_group_size` | 1e9 | Controls granularity of parameter partitioning |
| `stage3_prefetch_bucket_size` | "auto" | Prefetch buffer for parameters (bytes). Larger = more memory, less latency. |
| `stage3_param_persistence_threshold` | "auto" | Params smaller than this stay on GPU (bytes). Small params benefit from staying local. |
| `stage3_max_live_parameters` | 1e9 | Max params materialized simultaneously. Limits peak memory. |
| `stage3_max_reuse_distance` | 1e9 | Reuse threshold for gathered params. Larger = more caching, more memory. |
| `stage3_gather_16bit_weights_on_model_save` | false | Gather all params to rank 0 for saving. Required for standard checkpoint format. |

#### Offload Parameters

```json
{
  "zero_optimization": {
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true,
      "buffer_count": 4,
      "fast_init": false
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true,
      "buffer_count": 5,
      "buffer_size": 1e8,
      "max_in_cpu": 1e9
    }
  }
}
```

| Parameter | Description |
|-----------|-------------|
| `device` | "none", "cpu", or "nvme" |
| `pin_memory` | Pin CPU memory for faster transfers. Always true for offload. |
| `buffer_count` | Number of CPU buffers for async transfers |
| `buffer_size` | Size of each buffer (bytes) |
| `max_in_cpu` | Maximum params to keep in CPU (ZeRO-Infinity) |
| `fast_init` | Skip initialization for offloaded params |
| `nvme_path` | Path to NVMe storage (for nvme device) |

### Optimizer Configuration

DeepSpeed can manage the optimizer internally for better ZeRO integration.

```json
{
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-4,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.1
    }
  }
}
```

**Fused optimizers (faster):**
```json
{
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 3e-4,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.1,
      "adam_w_mode": true,
      "torch_adam": false
    }
  }
}
```

Setting `torch_adam: false` uses DeepSpeed's fused CUDA Adam, which is faster than PyTorch's.

### Learning Rate Scheduler

```json
{
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-4,
      "warmup_num_steps": 1000,
      "total_num_steps": 100000
    }
  }
}
```

**Available schedulers:**
- `WarmupLR` -- linear warmup, constant after
- `WarmupDecayLR` -- linear warmup, linear decay
- `WarmupCosineLR` -- linear warmup, cosine decay
- `OneCycle` -- one-cycle policy

### Activation Checkpointing

Trade compute for memory by recomputing activations during backward pass.

```json
{
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  }
}
```

| Parameter | Description |
|-----------|-------------|
| `partition_activations` | Partition activations across GPUs |
| `cpu_checkpointing` | Offload checkpointed activations to CPU |
| `contiguous_memory_optimization` | Use contiguous memory for checkpoints |
| `number_checkpoints` | Number of checkpoints (null = every layer) |

### Communication Configuration

```json
{
  "communication_data_type": "bf16",
  "prescale_gradients": false,
  "gradient_predivide_factor": 1.0,
  "sparse_gradients": false
}
```

### Logging and Monitoring

```json
{
  "tensorboard": {
    "enabled": true,
    "output_path": "/logs/tensorboard/",
    "job_name": "training_run"
  },
  "wandb": {
    "enabled": true,
    "project": "my-project",
    "team": "my-team"
  },
  "csv_monitor": {
    "enabled": true,
    "output_path": "/logs/csv/",
    "job_name": "training_run"
  }
}
```

## Production-Ready Configs

### 7B Model on 8x A100 (ZeRO-2)

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  },
  "gradient_clipping": 1.0,
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 3e-4,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.1,
      "torch_adam": false
    }
  },
  "scheduler": {
    "type": "WarmupCosineLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-4,
      "warmup_num_steps": 2000,
      "total_num_steps": 100000
    }
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true
  }
}
```

### 70B Model on 8x H100 (ZeRO-3)

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "none"},
    "offload_param": {"device": "none"},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_clipping": 1.0,
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 8,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1.5e-4,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.1,
      "torch_adam": false
    }
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true
  }
}
```

### 70B Model on 4x A100 with CPU Offload (Memory-Constrained)

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_clipping": 1.0,
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8,
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": true
  }
}
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| OOM at step 0 | Model too large for ZeRO stage | Increase stage (2->3) or add offload |
| OOM after first step | Optimizer states materialized | Increase stage or reduce batch size |
| Slow training with ZeRO-3 | High communication overhead | Increase bucket sizes, enable overlap_comm |
| Slow with CPU offload | PCIe bandwidth bottleneck | Use NVMe offload or more GPUs |
| `stage3_gather_16bit_weights_on_model_save` OOM | All params gathered to rank 0 | Save sharded checkpoint instead |
| Gradient NaN | Loss scaling issue | Switch bf16 (no loss scaling needed) |
| Hang during init | NCCL timeout | Increase NCCL_TIMEOUT, check network |
| Different results with different GPU counts | Batch size changed | Keep global batch size constant |
