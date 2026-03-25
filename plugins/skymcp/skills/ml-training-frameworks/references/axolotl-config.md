# Axolotl Configuration Reference

Complete YAML configuration reference for Axolotl v0.8.x.

## Model Configuration

```yaml
# Base model (HuggingFace model ID or local path)
base_model: meta-llama/Meta-Llama-3-8B-Instruct

# Model type (usually auto-detected)
model_type: LlamaForCausalLM

# Tokenizer (defaults to base_model)
tokenizer_type: AutoTokenizer
tokenizer_config: null

# Trust remote code (for custom models)
trust_remote_code: false

# Revision/branch
revision: main

# Model dtype
torch_dtype: bfloat16
```

## Adapter Configuration

### No Adapter (Full Fine-Tune)

```yaml
# Omit adapter field entirely for full fine-tune
```

### LoRA

```yaml
adapter: lora
lora_r: 32                    # Rank (8, 16, 32, 64, 128)
lora_alpha: 64                # Scaling factor (typically 2x rank)
lora_dropout: 0.05            # Dropout on LoRA layers
lora_target_linear: true      # Apply to all linear layers
lora_target_modules:          # OR specify explicitly
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
lora_fan_in_fan_out: false    # True for Conv1D (GPT-2 style)
```

### QLoRA (4-bit Quantized LoRA)

```yaml
adapter: qlora
load_in_4bit: true
bnb_4bit_quant_type: nf4      # nf4 or fp4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true

# LoRA params same as above
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_linear: true
```

### QAT (Quantization-Aware Training)

```yaml
adapter: lora
qat: true
qat_bits: 4
```

## Dataset Configuration

### Single Dataset

```yaml
datasets:
  - path: tatsu-lab/alpaca
    type: alpaca
```

### Multiple Datasets with Mixing

```yaml
datasets:
  - path: tatsu-lab/alpaca
    type: alpaca
    weight: 0.5               # Sampling weight
  - path: Open-Orca/OpenOrca
    type: sharegpt
    weight: 0.3
  - path: my_custom_data.jsonl
    type: completion
    weight: 0.2
```

### Dataset Types

| Type | Expected Columns | Description |
|------|-----------------|-------------|
| `alpaca` | instruction, input, output | Standard instruction format |
| `sharegpt` | conversations | Multi-turn with from/value |
| `completion` | text | Raw text completion |
| `input_output` | input, output | Simple pairs |
| `oasst` | text, role, lang | OpenAssistant format |
| `llava` | conversations, images | Vision-language pairs |

### Chat Templates

```yaml
datasets:
  - path: my_data.jsonl
    type: sharegpt
    conversation: chatml       # chatml, llama3, mistral, vicuna, zephyr
```

### Custom Prompt Templates

```yaml
datasets:
  - path: my_data.jsonl
    type: alpaca
    field_instruction: question
    field_input: context
    field_output: answer
```

### Local Files

```yaml
datasets:
  - path: /data/train.jsonl     # Absolute path
    type: alpaca
    ds_type: json               # json, parquet, csv, arrow
  - path: ./data/train.parquet  # Relative path
    type: completion
    ds_type: parquet
```

## Sequence Configuration

```yaml
sequence_len: 4096              # Max sequence length
sample_packing: true            # Pack multiple samples per sequence
pad_to_sequence_len: true       # Pad short sequences
```

### Sample Packing

When `sample_packing: true`, Axolotl concatenates multiple short samples into a single sequence to maximize GPU utilization. This can give 2-5x throughput improvement.

**Important**: Set `eval_sample_packing: false` for evaluation to get accurate per-sample metrics.

## Training Configuration

```yaml
# Batch sizes
micro_batch_size: 2             # Per-GPU batch size
gradient_accumulation_steps: 4  # Effective batch = micro * accum * num_gpus

# Training duration
num_epochs: 3                   # Number of epochs
max_steps: null                 # Override epochs with step count

# Optimizer
optimizer: adamw_bnb_8bit       # adamw_torch, adamw_bnb_8bit, lion_8bit, paged_adamw_8bit
lr_scheduler: cosine            # cosine, linear, constant, constant_with_warmup
learning_rate: 2e-4
weight_decay: 0.01
warmup_steps: 100               # OR warmup_ratio: 0.03
max_grad_norm: 1.0              # Gradient clipping

# Precision
bf16: auto                      # auto, true, false
tf32: true                      # TF32 tensor cores (A100+)
fp16: false                     # Not recommended if bf16 available

# Flash Attention
flash_attention: true           # Use FlashAttention-2/3
sdp_attention: false            # PyTorch scaled_dot_product_attention

# Gradient checkpointing
gradient_checkpointing: true    # Trade compute for memory
gradient_checkpointing_kwargs:
  use_reentrant: false
```

## Distributed Training

### FSDP2

```yaml
fsdp:
  - full_shard                  # full_shard, shard_grad_op, no_shard
  - auto_wrap
fsdp_config:
  fsdp_limit_all_gathers: true
  fsdp_sync_module_states: true
  fsdp_offload_params: false
  fsdp_use_orig_params: true
  fsdp_cpu_ram_efficient_loading: true
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_state_dict_type: FULL_STATE_DICT
```

### DeepSpeed (Alternative)

```yaml
deepspeed: configs/ds_config_zero3.json
```

## Logging and Saving

```yaml
# Weights & Biases
wandb_project: my-project
wandb_entity: my-team
wandb_name: run-name
wandb_log_model: false

# Logging frequency
logging_steps: 10
eval_steps: 500
eval_strategy: steps

# Checkpointing
save_strategy: steps
save_steps: 500
save_total_limit: 3
save_on_each_node: false

# Output
output_dir: /checkpoints
hub_model_id: myorg/my-model    # Push to HuggingFace Hub
push_to_hub: false
```

## Evaluation

```yaml
eval_strategy: steps
eval_steps: 500
eval_sample_packing: false      # Disable packing for eval
val_set_size: 0.05              # 5% of data for validation
# OR
datasets:
  - path: train.jsonl
    type: alpaca
    split: train
  - path: eval.jsonl
    type: alpaca
    split: eval
```

## Advanced Features

### Multipack (Efficient Packing)

```yaml
sample_packing: true
multipack_real_batches: true    # Allocate real batch dimension
sample_packing_eff_est: 3.0    # Estimated packing efficiency
```

### Curriculum Learning

```yaml
datasets:
  - path: easy_data.jsonl
    type: alpaca
    weight: 1.0
    # Processed first due to ordering
  - path: hard_data.jsonl
    type: alpaca
    weight: 0.0
    # Enabled later via callback
```

### NEFTune (Noise Embedding)

```yaml
neftune_noise_alpha: 5.0        # Add noise to embeddings during training
```

### Reward Model Training

```yaml
reward_model: true
datasets:
  - path: Anthropic/hh-rlhf
    type: chatml.default
```

### GRPO (Reasoning Post-Training)

```yaml
rl: grpo
rl_config:
  reward_model: meta-llama/Meta-Llama-3-8B-RM
  num_generations: 4
  kl_coeff: 0.05
  temperature: 0.7
  max_new_tokens: 2048
```

## Complete Production Example

```yaml
# Production Llama-3-8B QLoRA Fine-Tune
base_model: meta-llama/Meta-Llama-3-8B-Instruct
model_type: LlamaForCausalLM

adapter: qlora
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true

lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_linear: true

datasets:
  - path: /data/training/sft_data.jsonl
    type: sharegpt
    conversation: llama3

dataset_prepared_path: /cache/prepared
val_set_size: 0.02

sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true
eval_sample_packing: false

micro_batch_size: 2
gradient_accumulation_steps: 8
num_epochs: 2
optimizer: paged_adamw_8bit
lr_scheduler: cosine
learning_rate: 1e-4
warmup_steps: 50
weight_decay: 0.01
max_grad_norm: 1.0

bf16: auto
tf32: true
flash_attention: true
gradient_checkpointing: true

fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer

wandb_project: production-finetune
logging_steps: 5
save_strategy: steps
save_steps: 200
save_total_limit: 3
eval_strategy: steps
eval_steps: 200
output_dir: /checkpoints/llama3-8b-sft

neftune_noise_alpha: 5.0
```

## Launch Commands

```bash
# Single GPU
accelerate launch -m axolotl.cli.train config.yaml

# Multi-GPU (auto-detect)
accelerate launch -m axolotl.cli.train config.yaml

# Specific GPU count
accelerate launch --num_processes 8 -m axolotl.cli.train config.yaml

# With DeepSpeed
accelerate launch --config_file ds_config.yaml -m axolotl.cli.train config.yaml

# Preprocessing only (cache tokenized data)
python -m axolotl.cli.preprocess config.yaml

# Merge LoRA into base model
python -m axolotl.cli.merge_lora config.yaml --lora_model_dir /checkpoints/latest
```
