---
name: checkpoint-management
description: Use when saving or resuming training checkpoints, merging LoRA adapters, converting between model formats (safetensors, GGUF, PyTorch), quantizing models, using mergekit for model merging, or managing checkpoint lifecycle on cloud storage
---

# Checkpoint and Model Management

## Overview

Manage the full lifecycle of model artifacts: save checkpoints during training, resume across preemptions, merge adapters, convert between formats, and quantize for deployment. SkyPilot's persistent storage and stable task IDs make checkpoint management seamless across spot preemptions.

**Core principle:** Checkpoints are your insurance policy. Save early, save often, and always write to persistent storage outside the VM.

## When to Use

- Training any model (checkpoint saving and resumption)
- Recovering from spot preemption
- Merging LoRA/QLoRA adapters into base model
- Converting models between formats (HF, GGUF, SafeTensors)
- Quantizing models for deployment
- Merging multiple fine-tuned models with mergekit

**Do not use for:**
- Inference-only deployments (see serving skills)
- Dataset management (separate concern)

## SkyPilot Checkpoint Pattern

Use `MOUNT` mode for persistent checkpoint storage that survives preemptions.

```yaml
file_mounts:
  /checkpoints:
    name: my-ckpts
    store: s3
    mode: MOUNT

run: |
  python train.py \
    --checkpoint-dir /checkpoints/${SKYPILOT_TASK_ID} \
    --resume-from-latest
```

**Key details:**
- `MOUNT` mode: read/write cloud storage as local filesystem
- `MOUNT_CACHED`: caches reads locally for faster access (good for resuming)
- `SKYPILOT_TASK_ID` is stable across preemption recoveries -- same ID on resume
- Use `SKYPILOT_TASK_ID` for W&B run ID, checkpoint directory, and log paths

**Resume pattern:**
```python
import os
import glob

def find_latest_checkpoint(checkpoint_dir):
    """Find latest checkpoint for resumption."""
    pattern = os.path.join(checkpoint_dir, "checkpoint-*")
    checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime)
    if checkpoints:
        return checkpoints[-1]
    return None

checkpoint_dir = f"/checkpoints/{os.environ['SKYPILOT_TASK_ID']}"
latest = find_latest_checkpoint(checkpoint_dir)

if latest:
    print(f"Resuming from {latest}")
    model.load_state_dict(torch.load(os.path.join(latest, "model.safetensors")))
    optimizer.load_state_dict(torch.load(os.path.join(latest, "optimizer.pt")))
    step = int(latest.split("-")[-1])
else:
    print("Starting fresh")
    step = 0
```

## Async Checkpointing

Save checkpoints without stalling training. Copy tensors to CPU, write in background thread.

```python
import threading
import torch
import shutil

class AsyncCheckpointer:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self._thread = None

    def save(self, step, model, optimizer):
        # Copy state to CPU (fast, non-blocking)
        state = {
            "step": step,
            "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
        }

        # Wait for previous save to finish
        if self._thread is not None:
            self._thread.join()

        # Save in background
        def _write():
            path = os.path.join(self.checkpoint_dir, f"checkpoint-{step}")
            os.makedirs(path, exist_ok=True)
            torch.save(state["model"], os.path.join(path, "model.pt"))
            torch.save(state["optimizer"], os.path.join(path, "optimizer.pt"))
            # Atomic latest pointer
            latest_path = os.path.join(self.checkpoint_dir, "latest")
            with open(latest_path + ".tmp", "w") as f:
                f.write(str(step))
            os.replace(latest_path + ".tmp", latest_path)

        self._thread = threading.Thread(target=_write)
        self._thread.start()
```

**NeMo distributed checkpointing** (for large-scale training):
- Fully Parallel Saving (FPS): each GPU rank saves its own shard
- Async saving: copies to CPU memory, writes in background while training continues
- Flexible resumption: resume across different TP/PP configurations
- Configured via NeMo YAML: `exp_manager.checkpoint_callback_params.async_saving: true`

## Model Formats

| Format | Extension | Use Case | Notes |
|--------|-----------|----------|-------|
| SafeTensors | `.safetensors` | HF standard, training, serving | Memory-mapped, no code execution, safe |
| GGUF | `.gguf` | llama.cpp, Ollama, local inference | Quantized, single-file, fast CPU/GPU inference |
| PyTorch | `.bin`, `.pt` | Legacy, training | Pickle-based (security risk), being phased out |
| ONNX | `.onnx` | Cross-framework deployment | TensorRT, OpenVINO, DirectML |

**Always prefer SafeTensors** for training and HF ecosystem. Use GGUF only for deployment.

## LoRA/QLoRA Adapter Management

### Save Adapter
LoRA adapters are tiny (~10-100MB vs multi-GB full models).

```python
# After training
model.save_pretrained("/checkpoints/my-lora-adapter")
# Saves: adapter_model.safetensors + adapter_config.json
```

### Merge Adapter into Base Model
Required before GGUF conversion or non-PEFT serving.

```python
from peft import PeftModel, AutoModelForCausalLM

# Load base model
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load and merge adapter
model = PeftModel.from_pretrained(base, "/checkpoints/my-lora-adapter")
merged = model.merge_and_unload()

# Save merged model
merged.save_pretrained("/checkpoints/merged-model", safe_serialization=True)
```

### Multi-Adapter Stacking
Apply multiple LoRA adapters sequentially.

```python
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model = PeftModel.from_pretrained(base, "/adapters/sft-adapter")
model.load_adapter("/adapters/dpo-adapter", adapter_name="dpo")
model.set_adapter("dpo")  # activate DPO adapter on top of SFT
```

## Model Merging with mergekit

Combine multiple fine-tuned models into one without retraining.

```bash
pip install mergekit
```

### SLERP (Spherical Linear Interpolation)
Blend two models. Best for combining SFT + DPO or two fine-tunes of the same base.

```yaml
# merge_config.yaml
slices:
  - sources:
      - model: /models/sft-model
        layer_range: [0, 32]
      - model: /models/dpo-model
        layer_range: [0, 32]
merge_method: slerp
base_model: /models/sft-model
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5    # default
dtype: bfloat16
```

```bash
mergekit-yaml merge_config.yaml /output/merged-model --cuda
```

### DARE-TIES (Drop And REscale + TIES)
Merge 3+ models by pruning redundant parameter deltas.

```yaml
models:
  - model: /models/base
    parameters:
      weight: 1.0
  - model: /models/finetune-a
    parameters:
      weight: 0.5
      density: 0.5    # keep 50% of delta weights
  - model: /models/finetune-b
    parameters:
      weight: 0.5
      density: 0.5
merge_method: dare_ties
base_model: /models/base
parameters:
  int8_mask: true
dtype: bfloat16
```

### Merge Method Selection

| Method | Models | Best For |
|--------|--------|----------|
| SLERP | 2 | Blending SFT + DPO, combining complementary strengths |
| Linear | 2+ | Simple averaging, quick experiments |
| DARE-TIES | 3+ | Multiple specialized fine-tunes, pruning redundant deltas |
| TIES | 3+ | Similar to DARE-TIES, without random dropping |
| Passthrough | 2 | Franken-models: take layers from different models |

## GGUF Conversion and Quantization

Pipeline: Train (safetensors) -> Merge LoRA -> Convert to GGUF -> Quantize

```bash
# Clone llama.cpp for conversion tools
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Convert HF model to GGUF (FP16)
python convert_hf_to_gguf.py /path/to/merged-model --outfile model-f16.gguf --outtype f16

# Quantize with importance matrix (best quality)
# First, generate importance matrix from calibration data
./llama-imatrix -m model-f16.gguf -f calibration.txt -o imatrix.dat

# Then quantize using importance matrix
./llama-quantize --imatrix imatrix.dat model-f16.gguf model-Q4_K_M.gguf Q4_K_M
```

### Quantization Type Selection

| Type | Bits | Size (7B) | Quality | Use Case |
|------|------|-----------|---------|----------|
| Q2_K | 2.6 | ~2.8 GB | Low | Extreme compression, mobile |
| Q3_K_M | 3.4 | ~3.3 GB | Fair | Memory-constrained |
| Q4_K_M | 4.8 | ~4.4 GB | Good | **Default recommendation** |
| Q5_K_M | 5.7 | ~5.1 GB | Very Good | Quality-focused |
| Q6_K | 6.6 | ~5.9 GB | Near-FP16 | When quality matters most |
| Q8_0 | 8.5 | ~7.7 GB | Excellent | Minimal quality loss |
| F16 | 16 | ~14 GB | Lossless | Reference, no quantization |

**Rule of thumb:** Start with Q4_K_M. Use imatrix for best quality at given size.

## Full Pipeline: QLoRA Train to GGUF Deploy

```bash
# 1. QLoRA training (saves adapter)
python train_qlora.py \
  --base_model meta-llama/Llama-3.1-8B \
  --output_dir /checkpoints/my-adapter

# 2. Merge adapter into base
python merge_adapter.py \
  --base_model meta-llama/Llama-3.1-8B \
  --adapter /checkpoints/my-adapter \
  --output /checkpoints/merged-fp16

# 3. Convert to GGUF FP16
python llama.cpp/convert_hf_to_gguf.py \
  /checkpoints/merged-fp16 \
  --outfile /output/model-f16.gguf --outtype f16

# 4. Generate importance matrix
./llama.cpp/llama-imatrix \
  -m /output/model-f16.gguf \
  -f calibration.txt \
  -o /output/imatrix.dat

# 5. Quantize
./llama.cpp/llama-quantize \
  --imatrix /output/imatrix.dat \
  /output/model-f16.gguf \
  /output/model-Q4_K_M.gguf Q4_K_M

# 6. Test
./llama.cpp/llama-cli -m /output/model-Q4_K_M.gguf -p "Hello, world"
```

For detailed checkpoint strategies across distributed training setups, see `references/checkpoint-patterns.md`. For complete format conversion recipes, see `references/model-conversion.md`.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Saving to local disk on spot instances | Always save to mounted cloud storage (`/checkpoints` with MOUNT) |
| Not using SKYPILOT_TASK_ID | Without it, preempted jobs resume with a new ID and cannot find old checkpoints |
| Merging LoRA before saving base separately | Always keep the unmerged base + adapter. Merging is lossy for future adapter stacking. |
| GGUF without importance matrix | Quality degrades at low bit widths. Always use imatrix for Q4 and below. |
| Forgetting optimizer state in checkpoint | Model weights alone are not enough for resuming training. Save optimizer and scheduler too. |
| Non-atomic checkpoint writes | Power loss during save = corrupted checkpoint. Write to temp dir, then atomic rename. |

## Quick Reference

```bash
# Resume from latest checkpoint
python train.py --resume /checkpoints/${SKYPILOT_TASK_ID}/latest

# Merge LoRA adapter
python -c "
from peft import PeftModel, AutoModelForCausalLM
import torch
base = AutoModelForCausalLM.from_pretrained('MODEL', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, 'ADAPTER_PATH')
model.merge_and_unload().save_pretrained('OUTPUT', safe_serialization=True)
"

# Convert to GGUF
python convert_hf_to_gguf.py MODEL_DIR --outfile out.gguf --outtype f16

# Quantize
./llama-quantize --imatrix imatrix.dat out.gguf out-Q4_K_M.gguf Q4_K_M

# mergekit SLERP merge
mergekit-yaml config.yaml /output --cuda
```
