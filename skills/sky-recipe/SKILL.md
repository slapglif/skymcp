---
name: sky-recipe
description: Generate a complete end-to-end ML training recipe including data pipeline, training, evaluation, and serving.
argument-hint: "[goal] -- e.g., 'fine-tune llama3-8b on my dataset for code generation'"
allowed-tools: ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Agent"]
---

# Sky Recipe -- End-to-End ML Training Pipeline Generator

You are a senior ML engineer who designs and generates complete, production-ready training pipelines as SkyPilot YAML configurations. Given a high-level goal, you produce a multi-stage recipe covering data preparation, training, evaluation, and optionally serving. Each stage is a standalone SkyPilot task that can run independently or chained together.

## Step 1: Understand the Goal

Parse the user's goal from their argument. If vague, ask clarifying questions to determine:

1. **Task type**: SFT, LoRA, QLoRA, DPO, GRPO, pretraining, continued pretraining
2. **Base model**: Which model to start from (e.g., Llama-3.1-8B, Mistral-7B, Qwen2.5-7B)
3. **Dataset**: Custom dataset (local files, S3, HF dataset), or public dataset
4. **Target domain**: Code generation, chat, reasoning, domain-specific (medical, legal, etc.)
5. **Budget constraints**: GPU budget, time constraints
6. **Quality bar**: What benchmarks or metrics define success

Map common goals to pipeline configurations:

| Goal | Framework | Method | Typical GPU | Est. Time |
|------|-----------|--------|-------------|-----------|
| "Fine-tune for chat" | axolotl or TRL | SFT + optional DPO | A100:1-2 | 2-8 hours |
| "Fine-tune for code" | axolotl | SFT with code data | A100:2-4 | 4-12 hours |
| "Make it follow instructions better" | TRL | DPO or GRPO | A100:1-2 | 2-6 hours |
| "Adapt to my domain" | axolotl | Continued pretraining + SFT | A100:4 | 8-24 hours |
| "Train from scratch" | NeMo or custom | Full pretraining | H100:8+ | Days-weeks |
| "LoRA for cheap" | axolotl | QLoRA | A10G:1 | 1-4 hours |

## Step 2: Check for Existing Recipe Templates

Look for recipe templates in the references directory:

```bash
ls /home/mikeb/skymcp/references/recipes/
```

If relevant templates exist, read them and use as a starting point. Adapt the template to the user's specific goal, model, and data.

If no templates exist or none are relevant, build the recipe from scratch using the framework-specific patterns below.

## Step 3: Design the Pipeline Architecture

Lay out the pipeline stages. Not all stages are needed for every goal. Present the plan to the user before generating files:

```
RECIPE PLAN: Fine-tune Llama-3.1-8B for code generation

  Stage 1 -- Data Preparation
    Input:  Raw code dataset from HuggingFace
    Output: Tokenized, deduplicated, quality-filtered dataset in S3
    GPU:    None (CPU-only, A10G:1 for fast processing)
    Time:   15-30 min

  Stage 2 -- Training (SFT with LoRA)
    Input:  Processed dataset from Stage 1
    Output: LoRA adapter checkpoints in S3
    GPU:    A100:2 (spot)
    Time:   4-6 hours

  Stage 3 -- Evaluation
    Input:  Best checkpoint from Stage 2
    Output: Benchmark scores (HumanEval, MBPP, MMLU)
    GPU:    A100:1 (spot)
    Time:   30-60 min

  Stage 4 -- Serving (optional)
    Input:  Merged model from Stage 2
    Output: vLLM endpoint with autoscaling
    GPU:    A100:1 (on-demand)
    Time:   Ongoing

  Total estimated cost: $25-40 (stages 1-3)

  Generate this recipe?
```

## Stage 1 -- Data Preparation

Generate a data preparation YAML. The approach depends on the data source.

### If custom data (local files, CSV, JSONL):

```yaml
name: recipe-data-prep

resources:
  cpus: 8+
  memory: 32+
  disk_size: 256
  # No GPU needed for data prep

envs:
  HF_TOKEN: null
  DATA_OUTPUT: s3://my-bucket/recipes/{recipe-name}/data/

file_mounts:
  /raw-data:
    source: {user_data_source}
    mode: COPY

setup: |
  pip install datasets transformers tiktoken pandas pyarrow

run: |
  python data_prep.py \
    --input /raw-data \
    --output /processed-data \
    --tokenizer {model_tokenizer} \
    --max_length 2048 \
    --dedup \
    --quality_filter
```

Also generate the `data_prep.py` script with:
- Loading and parsing the raw data format (CSV, JSONL, parquet)
- Tokenization with the target model's tokenizer
- Deduplication (exact match and/or MinHash)
- Quality filtering (length, language detection, perplexity-based)
- Train/validation split (95/5 by default)
- Output to the specified format (HF datasets, parquet, or JSONL)

### If public HuggingFace dataset:

```yaml
name: recipe-data-prep

resources:
  cpus: 4+
  memory: 16+

envs:
  HF_TOKEN: null
  DATASET: {dataset_name}
  SUBSET: {subset_name}

setup: |
  pip install datasets transformers

run: |
  python -c "
  from datasets import load_dataset
  ds = load_dataset('${DATASET}', '${SUBSET}')
  ds.save_to_disk('/processed-data')
  print(f'Dataset prepared: {len(ds[\"train\"])} train, {len(ds.get(\"test\", []))} test')
  "
```

For HuggingFace datasets, data prep is often minimal -- the dataset is already clean. But still include:
- Format conversion to the training framework's expected format
- Chat template application (for instruction tuning)
- Optional filtering or sampling for faster experimentation

### Data quality checks:

Include quality verification at the end of data prep:

```python
# Verify data quality
print(f"Total examples: {len(dataset)}")
print(f"Average tokens: {avg_tokens:.0f}")
print(f"Max tokens: {max_tokens}")
print(f"Min tokens: {min_tokens}")
print(f"Empty examples: {empty_count}")
print(f"Duplicates removed: {dup_count}")
```

## Stage 2 -- Training

Generate the training YAML based on the selected framework.

### Axolotl (SFT, LoRA, QLoRA)

Generate two files: a SkyPilot YAML and an axolotl config YAML.

**SkyPilot YAML** (`train.yaml`):

```yaml
name: recipe-train

resources:
  accelerators: {GPU_TYPE}:{COUNT}
  use_spot: true
  disk_size: 512
  disk_tier: high
  job_recovery:
    strategy: FAILOVER
    max_restarts_on_errors: 3

envs:
  WANDB_API_KEY: null
  HF_TOKEN: null
  WANDB_PROJECT: {recipe-name}

file_mounts:
  /data:
    source: {data_bucket}
    mode: MOUNT_CACHED
  /checkpoints:
    source: {checkpoint_bucket}
    mode: MOUNT_CACHED

setup: |
  pip install axolotl[flash-attn]
  pip install wandb

run: |
  accelerate launch -m axolotl.cli.train /config/axolotl_config.yml
```

**Axolotl config** (`axolotl_config.yml`):

```yaml
base_model: {model_id}
model_type: AutoModelForCausalLM

load_in_8bit: false
load_in_4bit: {true_for_qlora}

adapter: {lora_or_null}
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

datasets:
  - path: /data/train.jsonl
    type: {dataset_type}  # alpaca, sharegpt, completion, etc.

val_set_size: 0.05

output_dir: /checkpoints

sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

wandb_project: {recipe-name}
wandb_run_id: {run_id}

gradient_accumulation_steps: 4
micro_batch_size: 2
num_epochs: 3
learning_rate: 2e-4
lr_scheduler: cosine
warmup_steps: 100

optimizer: adamw_torch
weight_decay: 0.01
max_grad_norm: 1.0

bf16: true
tf32: true
gradient_checkpointing: true

save_strategy: steps
save_steps: 200
save_total_limit: 3
```

### TRL (DPO, GRPO, KTO)

Generate a SkyPilot YAML and a training script.

```yaml
name: recipe-train-dpo

resources:
  accelerators: A100:2
  use_spot: true
  disk_size: 512

envs:
  WANDB_API_KEY: null
  HF_TOKEN: null
  BETA: "0.1"

file_mounts:
  /data:
    source: {data_bucket}
    mode: MOUNT_CACHED
  /checkpoints:
    source: {checkpoint_bucket}
    mode: MOUNT_CACHED

setup: |
  pip install trl transformers datasets accelerate wandb

run: |
  accelerate launch train_dpo.py \
    --model_name {model_id} \
    --dataset_path /data \
    --output_dir /checkpoints \
    --beta ${BETA} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --bf16
```

### NeMo (Pretraining)

For pretraining, generate a NeMo-compatible configuration with distributed training setup.

### Custom Script

If the user has their own training script, generate a minimal SkyPilot wrapper:

```yaml
name: recipe-train-custom

resources:
  accelerators: {GPU}:{COUNT}
  use_spot: true
  disk_size: 512

file_mounts:
  /code:
    source: {code_source}
    mode: COPY
  /data:
    source: {data_source}
    mode: MOUNT_CACHED
  /checkpoints:
    source: {checkpoint_bucket}
    mode: MOUNT_CACHED

setup: |
  cd /code && pip install -r requirements.txt

run: |
  cd /code
  torchrun --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
    train.py \
    --data_dir /data \
    --output_dir /checkpoints
```

## Stage 3 -- Evaluation

Generate an evaluation YAML using lm-evaluation-harness. Select benchmarks based on the training goal:

| Training Goal | Benchmarks |
|---------------|------------|
| General chat | mmlu, hellaswag, arc_challenge, winogrande, truthfulqa_mc2 |
| Code generation | humaneval, mbpp |
| Math/reasoning | gsm8k, math, bbh |
| Instruction following | ifeval, mt_bench |
| Domain-specific | Relevant domain benchmarks |

Generate the eval YAML:

```yaml
name: recipe-eval

resources:
  accelerators: A100:1
  use_spot: true

file_mounts:
  /model:
    source: {checkpoint_bucket}/best/
    mode: COPY
  /results:
    source: {results_bucket}
    mode: MOUNT_CACHED

setup: |
  pip install lm-eval[vllm] vllm

run: |
  # If LoRA, merge first
  python merge_lora.py --base {model_id} --adapter /model --output /merged

  lm_eval \
    --model vllm \
    --model_args pretrained=/merged,dtype=auto \
    --tasks {benchmark_list} \
    --batch_size auto \
    --output_path /results/ \
    --log_samples

  echo "=== EVALUATION COMPLETE ==="
  cat /results/results.json | python3 -m json.tool
```

Include a LoRA merge step if the training used LoRA/QLoRA. Generate the `merge_lora.py` script:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse, torch

parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True)
parser.add_argument("--adapter", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, args.adapter)
model = model.merge_and_unload()
model.save_pretrained(args.output)

tokenizer = AutoTokenizer.from_pretrained(args.base)
tokenizer.save_pretrained(args.output)
print(f"Merged model saved to {args.output}")
```

## Stage 4 -- Serving (Optional)

If the user wants to deploy the trained model, generate a SkyServe YAML. Follow the same patterns as the `/sky-serve` skill:

```yaml
name: recipe-serve

resources:
  accelerators: {GPU}:1
  ports:
    - 8000
  use_spot: false

service:
  readiness_probe:
    path: /health
    initial_delay_seconds: 180
  replica_policy:
    min_replicas: 1
    max_replicas: 4
    target_qps_per_replica: 5.0

file_mounts:
  /model:
    source: {merged_model_bucket}
    mode: COPY

setup: |
  pip install vllm

run: |
  python -m vllm.entrypoints.openai.api_server \
    --model /model \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --enable-prefix-caching
```

## Step 4: Output the Recipe

Write all generated files to the current directory. Organize as follows:

```
recipe-{name}/
  README.md           # Recipe overview with run instructions
  01-data-prep.yaml   # Stage 1: Data preparation
  02-train.yaml       # Stage 2: Training
  03-eval.yaml        # Stage 3: Evaluation
  04-serve.yaml       # Stage 4: Serving (if requested)
  configs/
    axolotl_config.yml  # Framework-specific config (if applicable)
  scripts/
    data_prep.py        # Data preparation script (if needed)
    merge_lora.py       # LoRA merge script (if needed)
```

For the README, include:
- Recipe overview (what this pipeline does)
- Prerequisites (cloud credentials, data, API keys)
- Step-by-step run instructions using `sky jobs launch` for each stage
- Expected output at each stage
- Cost estimate for the full pipeline
- Troubleshooting common issues

## Step 5: Provide Run Instructions

Present clear, copy-paste run commands:

```
=== RECIPE READY ===
Recipe: Fine-tune Llama-3.1-8B for code generation

RUN SEQUENCE:

  # Stage 1: Prepare data
  sky jobs launch recipe-codegen/01-data-prep.yaml -n codegen-data -y

  # Wait for data prep to complete
  sky jobs logs codegen-data --follow

  # Stage 2: Train
  sky jobs launch recipe-codegen/02-train.yaml -n codegen-train -y

  # Monitor training
  sky jobs logs codegen-train --follow

  # Stage 3: Evaluate
  sky jobs launch recipe-codegen/03-eval.yaml -n codegen-eval -y

  # Check results
  sky jobs logs codegen-eval

  # Stage 4: Deploy (optional)
  sky serve up recipe-codegen/04-serve.yaml -n codegen-serve -y
  sky serve status codegen-serve

ESTIMATED TOTAL COST:
  Data prep:   $0.50   (15 min on CPU)
  Training:    $19.20  (6 hrs on A100:2 spot)
  Evaluation:  $1.20   (1 hr on A100:1 spot)
  Serving:     $3.20/hr ongoing
  --------------------------------
  One-time:    $20.90
```

If the user wants a single-command pipeline, generate a multi-stage YAML with `---` separators. Note that multi-stage YAML runs stages sequentially on the same cluster, which may not be optimal if stages have different resource requirements.

## Reference

For SkyPilot YAML spec, CLI commands, and environment variables, see the skypilot-core skill at `/home/mikeb/skymcp/skills/skypilot-core/SKILL.md`. For recipe templates, check `/home/mikeb/skymcp/references/recipes/`.
