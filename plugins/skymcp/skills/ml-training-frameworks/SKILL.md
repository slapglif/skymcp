---
name: ml-training-frameworks
description: Use when selecting a training framework, comparing NeMo vs Axolotl vs torchtune vs TRL vs DeepSpeed vs Megatron, choosing between FSDP and DeepSpeed, deciding how to fine-tune or pretrain a model, or configuring LoRA/QLoRA/full fine-tuning - the definitive framework selection guide for ML training at any scale
---

# ML Training Framework Selection

## Decision Matrix

| Use Case | Framework | Why |
|----------|-----------|-----|
| Pretraining 100B+ params | NeMo 2.0 / Megatron-LM | Highest MFU, 5 parallelism axes (TP/PP/DP/EP/CP), FP8/FP4 |
| Pretraining 1B-70B params | torchtune / Axolotl + FSDP2 | Portable, good torch.compile support, no vendor lock |
| Fine-tuning (SFT) | Axolotl | YAML-driven, widest model support, LoRA/QLoRA/full |
| Post-training (DPO/GRPO/PPO) | TRL v0.28+ | Standard HF ecosystem, SFTTrainer/DPOTrainer/GRPOTrainer |
| Memory-constrained (huge model, small GPU) | DeepSpeed ZeRO-3 + CPU offload | Unbeatable memory savings for giant models on limited VRAM |
| Inference serving | vLLM | PagedAttention, 2-24x throughput vs naive, continuous batching |
| Research / custom architectures | torchtune | PyTorch-native, deep compile integration, clean recipe system |
| Multi-modal fine-tuning | Axolotl | Vision + language support, multipack, sample packing |

## Quick Start by Scenario

### Scenario 1: Fine-tune Llama-3-8B with LoRA

Use **Axolotl**:

```yaml
# axolotl.yaml
base_model: meta-llama/Meta-Llama-3-8B
model_type: LlamaForCausalLM
load_in_4bit: true
adapter: qlora
lora_r: 32
lora_alpha: 64
lora_target_linear: true
dataset_prepared_path: last_run_prepared
datasets:
  - path: dataset.jsonl
    type: alpaca
sequence_len: 4096
micro_batch_size: 2
gradient_accumulation_steps: 4
num_epochs: 3
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 2e-4
bf16: auto
flash_attention: true
```

```bash
accelerate launch -m axolotl.cli.train axolotl.yaml
```

### Scenario 2: Pretrain 7B from Scratch

Use **torchtune** with FSDP2:

```bash
tune run --nproc_per_node 8 full_finetune_distributed \
  --config recipes/llama3/7B_full.yaml \
  model.compile=True \
  training.enable_activation_checkpointing=True
```

### Scenario 3: RLHF / DPO Post-Training

Use **TRL**:

```python
from trl import DPOTrainer, DPOConfig

config = DPOConfig(
    model_name_or_path="meta-llama/Llama-3-8B-SFT",
    learning_rate=5e-7,
    beta=0.1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    bf16=True,
)
trainer = DPOTrainer(config=config, train_dataset=dataset)
trainer.train()
```

### Scenario 4: Pretrain 175B+

Use **NeMo 2.0**:

```python
from nemo.collections.llm import GPTModel, MegatronStrategy
from nemo.lightning import NeMoTrainer

strategy = MegatronStrategy(
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=4,
    context_parallel_size=2,
    expert_model_parallel_size=1,
    sequence_parallel=True,
)
trainer = NeMoTrainer(strategy=strategy, max_steps=100000)
```

## Framework Deep Comparison

### NeMo 2.0

- **Scale**: 100B+ parameters, multi-node clusters
- **Backend**: Megatron Core (TP, PP, DP, EP, CP)
- **Config**: NeMo-Run (Pythonic, not YAML)
- **Checkpointing**: Distributed checkpointing with FPS (Federated Parallel Save) + async writes
- **Precision**: FP8, FP4 via TransformerEngine
- **Parallelism**: 5 axes -- tensor, pipeline, data, expert, context
- **Best for**: Production pretraining at scale with maximum MFU
- **Gotcha**: Heavy dependency tree, steep learning curve

### Axolotl v0.8.x

- **Scale**: 1B-70B fine-tuning, moderate pretraining
- **Backend**: FSDP2 + optional DeepSpeed
- **Config**: YAML-driven (single file configures everything)
- **Methods**: Full fine-tune, LoRA, QLoRA, QAT, GRPO for reasoning
- **Features**: Sample packing, multipack, chat templates, multimodal
- **Model support**: Llama, Mistral, Phi, Qwen, Gemma, Falcon, MPT, and more
- **Best for**: Practitioners who want maximum flexibility from a single YAML
- **Gotcha**: Fast-moving; pin versions in production

### torchtune

- **Scale**: 1B-70B, research and production
- **Backend**: PyTorch-native, FSDP2, deep torch.compile integration
- **Config**: YAML recipes with CLI overrides
- **Design**: Clean recipe abstraction (each use case is a recipe)
- **Features**: Knowledge distillation, quantization-aware training, DPO
- **Best for**: Research teams who want PyTorch-native without framework overhead
- **Gotcha**: Smaller model zoo than Axolotl; newer ecosystem

### TRL v0.28+

- **Scale**: Fine-tuning up to 70B
- **Backend**: HuggingFace Transformers + Accelerate
- **Trainers**: SFTTrainer, DPOTrainer, PPOTrainer, GRPOTrainer, RewardTrainer, ORPOTrainer
- **Config**: Python dataclasses or YAML
- **Integration**: Direct HuggingFace Hub push, W&B logging
- **Best for**: RLHF / preference alignment workflows
- **Gotcha**: PPO is memory-hungry; prefer DPO or GRPO when possible

### DeepSpeed

- **Scale**: Any (1B-1T+)
- **Backend**: ZeRO optimizer stages 1-3
- **Config**: JSON configuration file
- **ZeRO Stage 1**: Partition optimizer states (1.5x memory savings)
- **ZeRO Stage 2**: + partition gradients (4x savings)
- **ZeRO Stage 3**: + partition parameters (8x+ savings) with CPU/NVMe offload
- **Best for**: When VRAM is the bottleneck and model must fit
- **Gotcha**: CPU offload adds latency; profile before committing

### Megatron-LM (Standalone)

- **Scale**: 2B-462B parameters
- **MFU**: 47% on H100 (reference benchmark)
- **Features**: Reference training scripts, proven convergence recipes
- **Best for**: Teams who want battle-tested pretraining scripts
- **Gotcha**: Less flexible than NeMo; more manual configuration

### FSDP2 (PyTorch Native)

- **What**: Fully Sharded Data Parallel, DTensor-level sharding
- **Used by**: torchtune, Axolotl, TRL (via Accelerate)
- **Advantages**: Native PyTorch, no external framework dependency, torch.compile friendly
- **When to use**: Default choice for 1-8 node training; switch to Megatron for larger
- **Gotcha**: Pipeline parallelism requires Megatron; FSDP2 is data + tensor parallel only

## Parallelism Strategy Guide

| Nodes | GPUs Total | Strategy |
|-------|-----------|----------|
| 1 | 1-2 | FSDP2 or DeepSpeed ZeRO-2 |
| 1 | 4-8 | FSDP2 with activation checkpointing |
| 2-8 | 16-64 | FSDP2 + gradient checkpointing |
| 8-32 | 64-256 | Megatron TP + FSDP DP |
| 32+ | 256+ | Megatron TP + PP + DP + CP |

## Memory Estimation

Rule-of-thumb for model memory (bf16 training):

| Component | Per-Parameter Cost |
|-----------|-------------------|
| Parameters | 2 bytes (bf16) |
| Gradients | 2 bytes (bf16) |
| Optimizer (AdamW) | 8 bytes (fp32 moments) |
| Activations | ~2 bytes (varies with seq length) |
| **Total** | **~14 bytes per parameter** |

A 7B model requires ~98 GB for full fine-tuning. LoRA reduces this to ~16-24 GB (only adapter parameters need optimizer states).

## LoRA vs QLoRA vs Full Fine-Tune

| Method | Memory | Speed | Quality | When to Use |
|--------|--------|-------|---------|-------------|
| Full fine-tune | Highest (14B/param) | Slowest | Best | Unlimited compute budget |
| LoRA (r=32) | ~30% of full | 2-3x faster | 95-99% of full | Default for most tasks |
| QLoRA (4-bit + LoRA) | ~15% of full | 2x faster | 93-97% of full | Limited VRAM |
| QAT (quantize-aware) | ~40% of full | Slower | Best at inference | Deploying quantized |

## Framework Compatibility Matrix

| Feature | NeMo 2.0 | Axolotl | torchtune | TRL | DeepSpeed |
|---------|----------|---------|-----------|-----|-----------|
| LoRA | Yes | Yes | Yes | Yes | Yes |
| QLoRA | No | Yes | Yes | Yes | Yes |
| FSDP2 | No (Megatron) | Yes | Yes | Yes | No (own sharding) |
| torch.compile | Partial | Partial | Full | Partial | No |
| Flash Attention | Yes | Yes | Yes | Yes | Yes |
| Multi-node | Yes | Yes | Yes | Yes | Yes |
| Multimodal | Yes | Yes | Partial | Partial | Yes |
| RLHF/DPO | Yes | GRPO | DPO | Full suite | With TRL |

See [references/framework-details.md](references/framework-details.md) for configuration deep dives.
See [references/axolotl-config.md](references/axolotl-config.md) for complete Axolotl YAML reference.
See [references/deepspeed-config.md](references/deepspeed-config.md) for ZeRO configuration reference.
