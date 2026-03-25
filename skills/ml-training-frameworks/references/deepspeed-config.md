# DeepSpeed Configuration Reference

Complete JSON configuration reference for DeepSpeed ZeRO optimization.

## ZeRO Stage 1: Optimizer State Partitioning

Partitions optimizer states across GPUs. ~4x memory savings.

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",

  "zero_optimization": {
    "stage": 1,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8
  },

  "bf16": {
    "enabled": true
  },

  "gradient_clipping": 1.0,

  "wall_clock_breakdown": false
}
```

## ZeRO Stage 2: + Gradient Partitioning

Partitions both optimizer states and gradients. ~8x memory savings.

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",

  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8,
    "contiguous_gradients": true
  },

  "bf16": {
    "enabled": true
  },

  "gradient_clipping": 1.0
}
```

## ZeRO Stage 3: + Parameter Partitioning

Partitions everything (optimizer states, gradients, parameters). Memory scales linearly with GPU count.

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",

  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8,
    "contiguous_gradients": true,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "sub_group_size": 1e12,
    "stage3_gather_16bit_weights_on_model_save": true
  },

  "bf16": {
    "enabled": true
  },

  "gradient_clipping": 1.0
}
```

## ZeRO Stage 3 + CPU Offload

Maximum memory savings. Offloads optimizer states and parameters to CPU RAM.

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",

  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "reduce_scatter": true,
    "contiguous_gradients": true,

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
    },

    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_max_live_parameters": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true,
    "sub_group_size": 1e12
  },

  "bf16": {
    "enabled": true
  },

  "gradient_clipping": 1.0
}
```

## ZeRO Stage 3 + NVMe Offload

For when CPU RAM is also insufficient. Offloads to NVMe SSD.

```json
{
  "zero_optimization": {
    "stage": 3,

    "offload_optimizer": {
      "device": "nvme",
      "nvme_path": "/local_nvme",
      "pin_memory": true,
      "buffer_count": 4,
      "fast_init": false
    },

    "offload_param": {
      "device": "nvme",
      "nvme_path": "/local_nvme",
      "pin_memory": true,
      "buffer_count": 5,
      "buffer_size": 1e8,
      "max_in_cpu": 1e9
    },

    "aio": {
      "block_size": 1048576,
      "queue_depth": 8,
      "thread_count": 1,
      "single_submit": false,
      "overlap_events": true
    }
  }
}
```

## Precision Configuration

### BF16 (Recommended for A100/H100)

```json
{
  "bf16": {
    "enabled": true
  }
}
```

### FP16 (Older GPUs)

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

### FP32 (Debugging Only)

```json
{
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": false
  }
}
```

## Optimizer Configuration

### Fused AdamW (Fastest)

```json
{
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": "auto"
    }
  }
}
```

### 1-bit Adam (Communication Efficient)

```json
{
  "optimizer": {
    "type": "OneBitAdam",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01,
      "freeze_step": 400,
      "cuda_aware": false,
      "comm_backend_name": "nccl"
    }
  }
}
```

## Scheduler Configuration

```json
{
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto",
      "total_num_steps": "auto"
    }
  }
}
```

## Communication Configuration

```json
{
  "communication_data_type": "bf16",
  "comms_logger": {
    "enabled": false,
    "verbose": false,
    "prof_all": false
  }
}
```

## Activation Checkpointing

```json
{
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  }
}
```

## Logging and Debugging

```json
{
  "steps_per_print": 100,
  "wall_clock_breakdown": true,
  "dump_state": false,

  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true
  }
}
```

## Complete Production Config: 70B QLoRA

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",

  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "reduce_scatter": true,
    "contiguous_gradients": true,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_gather_16bit_weights_on_model_save": true
  },

  "bf16": {
    "enabled": true
  },

  "gradient_clipping": 1.0,

  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true
  },

  "steps_per_print": 50,
  "wall_clock_breakdown": false
}
```

## Integration with HuggingFace Accelerate

Create `accelerate_config.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: ds_config.json
  zero3_init_flag: true
  zero3_save_16bit_model: true
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 8
use_cpu: false
```

Launch:
```bash
accelerate launch --config_file accelerate_config.yaml train.py
```

## Integration with Axolotl

In your Axolotl YAML:
```yaml
deepspeed: /path/to/ds_config.json
```

Or inline:
```yaml
deepspeed:
  zero_optimization:
    stage: 3
    offload_optimizer:
      device: cpu
    offload_param:
      device: cpu
  bf16:
    enabled: true
  gradient_clipping: 1.0
```

## Memory Estimation by Stage

For a model with P parameters (in billions):

| Stage | GPU Memory (per GPU, 8 GPUs) | CPU Memory |
|-------|------------------------------|------------|
| 0 (none) | 14P GB | Minimal |
| 1 | 14P / 1.6 GB | Minimal |
| 2 | 14P / 4 GB | Minimal |
| 3 | 14P / 8 GB | 14P * 0.5 GB |
| 3 + CPU | ~2P GB (activations only) | 14P GB |

Example: 70B model on 8x A100-80GB
- Stage 0: 980 GB needed -> impossible on 8x80=640 GB
- Stage 2: 245 GB -> ~31 GB/GPU -> fits
- Stage 3: 122 GB -> ~15 GB/GPU -> comfortable
- Stage 3 + CPU: ~140 GB GPU + 980 GB CPU -> fits on single node

## Performance Tips

1. **overlap_comm: true** -- Overlap gradient communication with backward pass. Always enable.
2. **reduce_scatter: true** -- More efficient than allreduce for large models. Always enable.
3. **contiguous_gradients: true** -- Reduces memory fragmentation. Always enable.
4. **Pin memory** -- `pin_memory: true` for CPU offload speeds up CPU-GPU transfers by 2-3x.
5. **Buffer sizes** -- Reduce `reduce_bucket_size` and `allgather_bucket_size` if OOM during communication.
6. **stage3_param_persistence_threshold** -- Parameters smaller than this stay on GPU. Increase for speed, decrease for memory.
