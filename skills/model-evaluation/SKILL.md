---
name: model-evaluation
description: Use when evaluating model quality, running benchmarks, comparing checkpoints, selecting evaluation tasks, or preparing leaderboard submissions - covers lm-evaluation-harness, lighteval, HELM, benchmark selection, and SkyPilot eval job patterns
---

# Model Evaluation and Benchmarking

## Overview

Evaluate trained models against standardized benchmarks to measure quality, detect regressions, and compare checkpoints. Run evaluations on cloud GPUs via SkyPilot to avoid blocking local resources.

**Core principle:** Every checkpoint decision (keep/discard/deploy) requires quantitative evaluation against a known baseline. Never ship a model without benchmarking it.

## When to Use

- After training completes or a checkpoint is saved
- Comparing two model versions (A/B)
- Preparing a model for deployment or leaderboard submission
- Validating that fine-tuning did not degrade base capabilities
- Selecting which benchmarks matter for a given use case

**Do not use for:**
- Vibes-based evaluation (use MT-Bench or human eval instead)
- Latency/throughput testing (use inference benchmarking tools)
- Training loss curves (use training-monitoring skill)

## Primary Tools

### lm-evaluation-harness (EleutherAI)

The standard. Backend for the Open LLM Leaderboard. 200+ tasks, YAML config, chat template support.

```bash
# Install
pip install lm-eval

# Run evaluation
lm_eval --model hf \
  --model_args pretrained=/path/to/model \
  --tasks mmlu,hellaswag,arc_easy,arc_challenge,winogrande,gsm8k \
  --batch_size auto \
  --output_path /results/

# List available tasks
lm_eval --tasks list

# Use chat template for instruction-tuned models
lm_eval --model hf \
  --model_args pretrained=/path/to/model \
  --tasks mmlu \
  --apply_chat_template \
  --batch_size auto

# VLLM backend for faster inference
lm_eval --model vllm \
  --model_args pretrained=/path/to/model,tensor_parallel_size=2 \
  --tasks mmlu \
  --batch_size auto
```

**Key flags:**
- `--batch_size auto` -- auto-detect max batch for available VRAM
- `--num_fewshot N` -- override default few-shot count
- `--limit 100` -- run only 100 samples per task (fast debugging)
- `--log_samples` -- save per-sample predictions for error analysis
- `--apply_chat_template` -- required for chat/instruct models

### lighteval (HuggingFace)

Lighter weight, tighter HF Hub integration, faster iteration cycle.

```bash
pip install lighteval

# Run evaluation
lighteval accelerate \
  --model_args "pretrained=/path/to/model" \
  --tasks "leaderboard|mmlu|5" \
  --output_dir /results/

# Evaluate model directly from HF Hub
lighteval accelerate \
  --model_args "pretrained=meta-llama/Llama-3-8B" \
  --tasks "leaderboard|hellaswag|10"
```

**When to prefer lighteval:**
- Quick iteration during training (lighter overhead)
- HF Hub models (native integration)
- Custom task definitions (simpler YAML format)

### HELM (Stanford)

Holistic evaluation: accuracy, calibration, robustness, fairness, bias, toxicity, efficiency. Use when evaluation must cover more than accuracy.

```bash
pip install crfm-helm
helm-run --run-entries mmlu:model=hf/my-model --suite my-eval
helm-summarize --suite my-eval
```

**When to prefer HELM:**
- Safety-critical deployments
- Bias and fairness audits
- Multi-dimensional evaluation (not just accuracy)

## Benchmark Selection Guide

| Use Case | Benchmarks | Why |
|----------|-----------|-----|
| General knowledge | MMLU, ARC, HellaSwag, WinoGrande | Broad coverage of reasoning and knowledge |
| Math/reasoning | GSM8K, MATH, BBH | Chain-of-thought and multi-step |
| Code generation | HumanEval, MBPP, MultiPL-E | Functional correctness |
| Instruction following | MT-Bench, AlpacaEval, IFEval | Chat quality and compliance |
| Safety | TruthfulQA, ToxiGen, BBQ | Hallucination and toxicity |
| Long context | RULER, Needle-in-Haystack | Context window utilization |

For detailed benchmark descriptions and scoring methodology, see `references/benchmark-guide.md`.

## SkyPilot Eval Job Pattern

Run evaluation on cloud GPUs without blocking local machines.

```yaml
name: model-eval
resources:
  accelerators: A100:1
  disk_size: 200

file_mounts:
  /model:
    source: s3://my-checkpoints/latest/
  /results:
    name: eval-results
    store: s3
    mode: MOUNT

setup: |
  pip install lm-eval vllm

run: |
  lm_eval --model vllm \
    --model_args pretrained=/model,tensor_parallel_size=1 \
    --tasks mmlu,hellaswag,arc_easy,arc_challenge,winogrande,gsm8k \
    --batch_size auto \
    --output_path /results/${SKYPILOT_TASK_ID}/
```

For multi-GPU eval, VLLM backend recipes, and custom task configs, see `references/eval-recipes.md`.

## Post-Training Eval Pipeline

```
train checkpoint
      |
      v
  eval on standard suite
      |
      v
  compare to baseline
      |
      +---> better? KEEP, update baseline
      |
      +---> worse?  DISCARD, investigate
      |
      +---> mixed?  eval on domain-specific tasks, decide
```

**Automation pattern:**

```bash
# In training script callback or post-training step
BASELINE_MMLU=0.65
RESULT=$(lm_eval --model hf --model_args pretrained=/ckpt \
  --tasks mmlu --batch_size auto --output_path /tmp/eval/ \
  | grep "mmlu" | awk '{print $NF}')

if (( $(echo "$RESULT > $BASELINE_MMLU" | bc -l) )); then
  echo "KEEP: MMLU $RESULT > baseline $BASELINE_MMLU"
  cp -r /ckpt /checkpoints/promoted/
else
  echo "DISCARD: MMLU $RESULT <= baseline $BASELINE_MMLU"
fi
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Evaluating chat model without `--apply_chat_template` | Scores will be much lower than actual capability. Always use chat template for instruct models. |
| Using `--limit` in final eval | Limit is for debugging only. Full eval required for publication or decisions. |
| Comparing different few-shot counts | Always match `--num_fewshot` between runs. |
| Ignoring confidence intervals | Small differences (< 1-2%) may be noise. Run multiple seeds or use full eval sets. |
| Running on CPU | Evaluation is slow on CPU. Use GPU, especially with VLLM backend. |
| Not pinning lm-eval version | Different versions can produce different scores. Pin the version in requirements. |

## Quick Reference

```bash
# Standard eval suite (Open LLM Leaderboard v2 tasks)
lm_eval --model hf --model_args pretrained=MODEL \
  --tasks mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,gsm8k \
  --batch_size auto --output_path /results/

# Fast sanity check (100 samples per task)
lm_eval --model hf --model_args pretrained=MODEL \
  --tasks mmlu --limit 100 --batch_size auto

# Compare two checkpoints
lm_eval --model hf --model_args pretrained=CKPT_A \
  --tasks mmlu --output_path /results/a/
lm_eval --model hf --model_args pretrained=CKPT_B \
  --tasks mmlu --output_path /results/b/
diff <(cat /results/a/results.json) <(cat /results/b/results.json)

# List all tasks matching a pattern
lm_eval --tasks list | grep -i "math"
```
