# ML Training Framework Details

Deep dive into each framework's architecture, configuration, and production patterns.

## NeMo 2.0

### Architecture

NeMo 2.0 is built on Megatron Core, NVIDIA's production-grade distributed training library. It provides 5 parallelism axes:

| Axis | Abbreviation | What It Shards | When to Use |
|------|-------------|----------------|-------------|
| Tensor Parallel | TP | Attention heads, FFN columns | Always for large models |
| Pipeline Parallel | PP | Layers across GPUs | When model > single-node memory |
| Data Parallel | DP | Batches | Default, always use |
| Expert Parallel | EP | MoE experts | Mixture of Experts models |
| Context Parallel | CP | Sequence dimension | Very long sequences (>32K) |

### NeMo-Run Configuration (Pythonic, Not YAML)

```python
from nemo.collections.llm import GPTModel, PretrainConfig
from nemo.lightning import NeMoTrainer, MegatronStrategy

# Model configuration
model_config = GPTModel.Config(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    num_query_groups=8,  # GQA
    ffn_hidden_size=11008,
    max_position_embeddings=4096,
    vocab_size=32000,
    bf16=True,
    apply_rope=True,
)

# Parallelism strategy
strategy = MegatronStrategy(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=2,
    context_parallel_size=1,
    expert_model_parallel_size=1,
    sequence_parallel=True,
    gradient_as_bucket_view=True,
)

# Trainer
trainer = NeMoTrainer(
    strategy=strategy,
    max_steps=100000,
    val_check_interval=1000,
    log_every_n_steps=10,
    limit_val_batches=50,
    accumulate_grad_batches=2,
    gradient_clip_val=1.0,
)
```

### Distributed Checkpointing

NeMo 2.0 uses Federated Parallel Save (FPS) with async writes:

```python
from nemo.lightning.io import TrainerCheckpoint

checkpoint_config = TrainerCheckpoint(
    save_dir="/checkpoints",
    save_interval=500,      # Save every 500 steps
    save_top_k=3,           # Keep 3 best checkpoints
    async_save=True,        # Non-blocking checkpoint writes
    save_last=True,
)
```

Key advantage: Distributed checkpointing does NOT require all ranks to have the same model structure. You can resume a TP=4 checkpoint with TP=8.

### FP8 / FP4 Training

```python
from nemo.collections.llm import TransformerEngineConfig

te_config = TransformerEngineConfig(
    fp8=True,
    fp8_margin=0,
    fp8_interval=1,
    fp8_amax_history_len=1024,
    fp8_amax_compute_algo="max",
)
```

FP8 training gives ~1.5-2x speedup on H100/B200 with minimal quality loss.

## Axolotl v0.8.x

### Configuration System

Axolotl uses a single YAML file that configures everything:

```yaml
# Model
base_model: meta-llama/Meta-Llama-3-8B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer
trust_remote_code: true

# LoRA
adapter: lora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_linear: true
lora_fan_in_fan_out: false

# Quantization (QLoRA)
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16

# Dataset
datasets:
  - path: tatsu-lab/alpaca
    type: alpaca
  - path: my_data.jsonl
    type: sharegpt
    conversation: chatml

# Training
sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true
micro_batch_size: 2
gradient_accumulation_steps: 4
num_epochs: 3
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 2e-4
warmup_steps: 100
weight_decay: 0.01
max_grad_norm: 1.0

# Precision
bf16: auto
tf32: true
flash_attention: true

# Distributed
fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer

# Logging
wandb_project: my-project
wandb_entity: my-team
logging_steps: 10
save_strategy: steps
save_steps: 500

# Eval
eval_strategy: steps
eval_steps: 500
eval_sample_packing: false
```

### Dataset Formats

| Format | Description | Example |
|--------|-------------|---------|
| `alpaca` | instruction/input/output | `{"instruction": "...", "input": "...", "output": "..."}` |
| `sharegpt` | Multi-turn conversation | `{"conversations": [{"from": "human", "value": "..."}, ...]}` |
| `completion` | Raw text completion | `{"text": "..."}` |
| `oasst` | OpenAssistant format | `{"text": "...", "role": "...", "lang": "..."}` |
| `input_output` | Simple pairs | `{"input": "...", "output": "..."}` |

### GRPO for Reasoning (New in v0.8)

```yaml
# Axolotl GRPO config
rl: grpo
rl_config:
  reward_model: meta-llama/Meta-Llama-3-8B-RM
  num_generations: 4
  kl_coeff: 0.05
  temperature: 0.7
```

### Multimodal Fine-Tuning

```yaml
base_model: llava-hf/llava-v1.6-mistral-7b-hf
model_type: LlavaForConditionalGeneration
datasets:
  - path: my_images.jsonl
    type: llava
    image_dir: /data/images
```

## torchtune

### Recipe System

torchtune organizes training as "recipes" -- standalone training scripts:

| Recipe | Command | Use Case |
|--------|---------|----------|
| `full_finetune_single_device` | `tune run full_finetune_single_device` | Single GPU full fine-tune |
| `full_finetune_distributed` | `tune run --nproc_per_node 8 full_finetune_distributed` | Multi-GPU full fine-tune |
| `lora_finetune_single_device` | `tune run lora_finetune_single_device` | Single GPU LoRA |
| `lora_finetune_distributed` | `tune run --nproc_per_node 8 lora_finetune_distributed` | Multi-GPU LoRA |
| `knowledge_distillation_single_device` | `tune run knowledge_distillation_single_device` | KD on single GPU |
| `quantize` | `tune run quantize` | Post-training quantization |
| `generate` | `tune run generate` | Text generation |

### Configuration

```yaml
# recipes/llama3/8B_lora.yaml
model:
  _component_: torchtune.models.llama3.lora_llama3_8b
  lora_attn_modules: [q_proj, v_proj, k_proj, output_proj]
  lora_rank: 32
  lora_alpha: 64

tokenizer:
  _component_: torchtune.models.llama3.llama3_tokenizer
  path: /models/Meta-Llama-3-8B/tokenizer.model

dataset:
  _component_: torchtune.datasets.alpaca_cleaned_dataset
  max_seq_len: 4096

training:
  batch_size: 2
  epochs: 3
  max_steps_per_epoch: null
  gradient_accumulation_steps: 16
  optimizer:
    _component_: torch.optim.AdamW
    lr: 2e-4
    weight_decay: 0.01
  lr_scheduler:
    _component_: torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup
    num_warmup_steps: 100
  compile: true
  enable_activation_checkpointing: true

# Logging
metric_logger:
  _component_: torchtune.training.metric_logging.WandBLogger
  project: my-project

# Checkpointing
checkpointer:
  _component_: torchtune.training.FullModelHFCheckpointer
  checkpoint_dir: /models/Meta-Llama-3-8B
  output_dir: /checkpoints
  checkpoint_every_n_steps: 500
```

### torch.compile Integration

torchtune has the deepest torch.compile support of any framework:

```yaml
training:
  compile: true  # Enables torch.compile on the model
```

This can give 1.5-3x speedup depending on model and hardware. Works best on:
- H100 and newer GPUs
- Models without dynamic shapes
- Stable training (compile overhead amortizes over many steps)

## TRL v0.28+

### Trainer Classes

| Trainer | Purpose | Key Args |
|---------|---------|----------|
| `SFTTrainer` | Supervised fine-tuning | `dataset_text_field`, `packing`, `max_seq_length` |
| `DPOTrainer` | Direct Preference Optimization | `beta`, `loss_type` |
| `PPOTrainer` | Proximal Policy Optimization | `mini_batch_size`, `ppo_epochs`, `learning_rate` |
| `GRPOTrainer` | Group Relative Policy Opt | `num_generations`, `kl_coeff` |
| `RewardTrainer` | Reward model training | `max_length` |
| `ORPOTrainer` | Odds Ratio Preference Opt | `beta`, `max_length` |

### SFT Example

```python
from trl import SFTConfig, SFTTrainer

config = SFTConfig(
    model_name_or_path="meta-llama/Llama-3-8B",
    dataset_text_field="text",
    max_seq_length=4096,
    packing=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    output_dir="/checkpoints",
)

trainer = SFTTrainer(config=config, train_dataset=train_dataset)
trainer.train()
trainer.save_model("/models/finetuned")
```

### DPO Example

```python
from trl import DPOConfig, DPOTrainer

config = DPOConfig(
    model_name_or_path="meta-llama/Llama-3-8B-SFT",
    beta=0.1,                    # KL penalty strength
    loss_type="sigmoid",         # sigmoid, hinge, ipo
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    bf16=True,
    max_length=4096,
    max_prompt_length=2048,
    output_dir="/checkpoints",
)

# Dataset must have "prompt", "chosen", "rejected" columns
trainer = DPOTrainer(
    config=config,
    train_dataset=dpo_dataset,
)
trainer.train()
```

### GRPO Example (Reasoning)

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    model_name_or_path="meta-llama/Llama-3-8B-SFT",
    num_generations=4,           # Generate N responses per prompt
    kl_coeff=0.05,              # KL penalty
    temperature=0.7,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-6,
    output_dir="/checkpoints",
)

trainer = GRPOTrainer(
    config=config,
    train_dataset=prompts_dataset,
    reward_fn=my_reward_function,  # Custom reward
)
trainer.train()
```

## DeepSpeed

### ZeRO Stages

| Stage | What Is Sharded | Memory Savings | Overhead |
|-------|----------------|----------------|----------|
| 0 | Nothing | 1x | None |
| 1 | Optimizer states | 4x | Minimal |
| 2 | + Gradients | 8x | Low |
| 3 | + Parameters | Linear with GPU count | Medium |
| 3 + CPU offload | Everything, overflow to CPU | Maximum | High latency |

### Integration Points

DeepSpeed can be used with:
- Axolotl (via config YAML)
- TRL/Transformers (via Accelerate)
- Custom training loops (via `deepspeed.initialize()`)

## Megatron-LM (Standalone)

### When to Use

Use standalone Megatron-LM when:
- You need maximum control over the training loop
- NeMo's abstractions add unwanted overhead
- You are pretraining from scratch at 100B+ scale
- You want to modify Megatron Core internals

### Key Scripts

```bash
# Pretrain GPT
python pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    --num-layers 64 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --micro-batch-size 1 \
    --global-batch-size 2048 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --train-iters 100000 \
    --lr 1.5e-4 \
    --min-lr 1.5e-5 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --bf16 \
    --use-flash-attn \
    --data-path /data/megatron_data \
    --tokenizer-type GPT2BPETokenizer \
    --save /checkpoints \
    --save-interval 1000
```

### Performance Reference

| Model Size | GPUs | TP | PP | DP | MFU | Tokens/sec |
|-----------|------|----|----|----|----|-----------|
| 7B | 8xH100 | 1 | 1 | 8 | 52% | ~65K |
| 70B | 64xH100 | 8 | 1 | 8 | 47% | ~32K |
| 175B | 128xH100 | 8 | 4 | 4 | 43% | ~18K |
| 462B | 512xH100 | 8 | 8 | 8 | 40% | ~8K |
