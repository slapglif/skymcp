# Eval Recipes for SkyPilot

Ready-to-use SkyPilot YAML recipes for different evaluation scenarios. Copy, modify the model path, and launch.

## Recipe 1: Standard Eval Suite (Single GPU)

Basic evaluation on the common benchmark suite. Good for models up to ~13B parameters.

```yaml
name: eval-standard
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
  pip install lm-eval==0.4.6 vllm==0.6.6

run: |
  echo "Starting standard eval suite for ${SKYPILOT_TASK_ID}"

  lm_eval --model vllm \
    --model_args pretrained=/model,tensor_parallel_size=1,dtype=bfloat16 \
    --tasks mmlu,hellaswag,arc_easy,arc_challenge,winogrande,gsm8k,truthfulqa_mc2 \
    --batch_size auto \
    --output_path /results/${SKYPILOT_TASK_ID}/ \
    --log_samples

  echo "Eval complete. Results at /results/${SKYPILOT_TASK_ID}/"
```

**Launch:**
```bash
sky launch eval-standard.yaml --env MODEL_PATH=s3://my-checkpoints/latest/
```

## Recipe 2: Fast Sanity Check (Subset Sampling)

Quick validation that a checkpoint is not broken. Runs 100 samples per task. Completes in minutes instead of hours.

```yaml
name: eval-sanity
resources:
  accelerators: A100:1
  disk_size: 100

file_mounts:
  /model:
    source: s3://my-checkpoints/latest/

setup: |
  pip install lm-eval==0.4.6

run: |
  lm_eval --model hf \
    --model_args pretrained=/model,dtype=bfloat16 \
    --tasks mmlu,hellaswag,arc_challenge \
    --limit 100 \
    --batch_size auto \
    --output_path /tmp/sanity/

  echo "=== SANITY CHECK RESULTS ==="
  cat /tmp/sanity/results.json | python3 -c "
  import json, sys
  data = json.load(sys.stdin)
  for task, metrics in data.get('results', {}).items():
      acc = metrics.get('acc,none', metrics.get('acc_norm,none', 'N/A'))
      print(f'{task}: {acc}')
  "
```

## Recipe 3: Large Model Eval (Multi-GPU with VLLM)

For 30B-70B models that need tensor parallelism. Uses VLLM backend for fast inference.

```yaml
name: eval-large
resources:
  accelerators: A100-80GB:4
  disk_size: 500

file_mounts:
  /model:
    source: s3://my-checkpoints/llama-70b/
  /results:
    name: eval-results-large
    store: s3
    mode: MOUNT

setup: |
  pip install lm-eval==0.4.6 vllm==0.6.6

run: |
  lm_eval --model vllm \
    --model_args pretrained=/model,tensor_parallel_size=4,dtype=bfloat16,max_model_len=4096 \
    --tasks mmlu,hellaswag,arc_challenge,winogrande,gsm8k,truthfulqa_mc2 \
    --batch_size auto \
    --output_path /results/${SKYPILOT_TASK_ID}/ \
    --log_samples
```

## Recipe 4: Chat Model Eval (Instruction-Tuned)

Evaluates instruction-tuned models with chat template applied. Includes IFEval for instruction compliance.

```yaml
name: eval-chat
resources:
  accelerators: A100:1
  disk_size: 200

file_mounts:
  /model:
    source: s3://my-checkpoints/chat-model/
  /results:
    name: eval-chat-results
    store: s3
    mode: MOUNT

setup: |
  pip install lm-eval==0.4.6 vllm==0.6.6

run: |
  # Standard benchmarks with chat template
  lm_eval --model vllm \
    --model_args pretrained=/model,dtype=bfloat16 \
    --tasks mmlu,arc_challenge,hellaswag,winogrande,gsm8k \
    --apply_chat_template \
    --batch_size auto \
    --output_path /results/${SKYPILOT_TASK_ID}/standard/

  # Instruction following eval
  lm_eval --model vllm \
    --model_args pretrained=/model,dtype=bfloat16 \
    --tasks ifeval \
    --apply_chat_template \
    --batch_size auto \
    --output_path /results/${SKYPILOT_TASK_ID}/ifeval/
```

## Recipe 5: Code Model Eval

HumanEval and MBPP with pass@k sampling.

```yaml
name: eval-code
resources:
  accelerators: A100:1
  disk_size: 200

file_mounts:
  /model:
    source: s3://my-checkpoints/code-model/
  /results:
    name: eval-code-results
    store: s3
    mode: MOUNT

setup: |
  pip install lm-eval==0.4.6 vllm==0.6.6

run: |
  # pass@1 (greedy)
  lm_eval --model vllm \
    --model_args pretrained=/model,dtype=bfloat16 \
    --tasks humaneval,mbpp \
    --batch_size auto \
    --output_path /results/${SKYPILOT_TASK_ID}/pass1/ \
    --log_samples

  # pass@10 (temperature sampling)
  lm_eval --model vllm \
    --model_args pretrained=/model,dtype=bfloat16,temperature=0.8 \
    --tasks humaneval \
    --batch_size auto \
    --num_fewshot 0 \
    --output_path /results/${SKYPILOT_TASK_ID}/pass10/ \
    --log_samples
```

## Recipe 6: A/B Comparison (Two Checkpoints)

Run identical eval on two checkpoints and produce comparison output.

```yaml
name: eval-compare
resources:
  accelerators: A100:1
  disk_size: 300

file_mounts:
  /model_a:
    source: s3://my-checkpoints/baseline/
  /model_b:
    source: s3://my-checkpoints/candidate/
  /results:
    name: eval-compare-results
    store: s3
    mode: MOUNT

setup: |
  pip install lm-eval==0.4.6 vllm==0.6.6

run: |
  TASKS="mmlu,hellaswag,arc_challenge,winogrande,gsm8k,truthfulqa_mc2"

  echo "=== Evaluating Model A (baseline) ==="
  lm_eval --model vllm \
    --model_args pretrained=/model_a,dtype=bfloat16 \
    --tasks $TASKS \
    --batch_size auto \
    --output_path /results/${SKYPILOT_TASK_ID}/model_a/

  echo "=== Evaluating Model B (candidate) ==="
  lm_eval --model vllm \
    --model_args pretrained=/model_b,dtype=bfloat16 \
    --tasks $TASKS \
    --batch_size auto \
    --output_path /results/${SKYPILOT_TASK_ID}/model_b/

  echo "=== COMPARISON ==="
  python3 -c "
  import json, os

  def load_results(path):
      with open(os.path.join(path, 'results.json')) as f:
          return json.load(f)['results']

  a = load_results('/results/${SKYPILOT_TASK_ID}/model_a/')
  b = load_results('/results/${SKYPILOT_TASK_ID}/model_b/')

  print(f'{'Task':<25} {'Model A':>10} {'Model B':>10} {'Delta':>10}')
  print('-' * 55)
  for task in sorted(a.keys()):
      va = a[task].get('acc,none', a[task].get('acc_norm,none', 0))
      vb = b.get(task, {}).get('acc,none', b.get(task, {}).get('acc_norm,none', 0))
      delta = vb - va
      marker = '+' if delta > 0 else ''
      print(f'{task:<25} {va:>10.4f} {vb:>10.4f} {marker}{delta:>9.4f}')
  "
```

## Recipe 7: Lightweight Eval with lighteval

For HuggingFace Hub models or when you want lighter dependencies.

```yaml
name: eval-lighteval
resources:
  accelerators: A100:1
  disk_size: 200

file_mounts:
  /results:
    name: lighteval-results
    store: s3
    mode: MOUNT

setup: |
  pip install lighteval[accelerate]

run: |
  lighteval accelerate \
    --model_args "pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16" \
    --tasks "leaderboard|mmlu|5,leaderboard|hellaswag|10,leaderboard|arc:challenge|25" \
    --output_dir /results/${SKYPILOT_TASK_ID}/
```

## Recipe 8: Continuous Eval (Post-Training Callback)

Attach to a training job. Evaluates latest checkpoint every N steps.

```yaml
name: train-with-eval
resources:
  accelerators: A100:4
  disk_size: 500

file_mounts:
  /data:
    source: s3://my-data/
  /checkpoints:
    name: training-ckpts
    store: s3
    mode: MOUNT

setup: |
  pip install lm-eval==0.4.6 torch transformers

run: |
  # Train with periodic eval
  python3 train.py \
    --data_dir /data \
    --checkpoint_dir /checkpoints/${SKYPILOT_TASK_ID} \
    --eval_every 1000 &
  TRAIN_PID=$!

  # Watch for new checkpoints and eval
  LAST_CKPT=""
  while kill -0 $TRAIN_PID 2>/dev/null; do
    LATEST=$(ls -t /checkpoints/${SKYPILOT_TASK_ID}/checkpoint-* 2>/dev/null | head -1)
    if [ -n "$LATEST" ] && [ "$LATEST" != "$LAST_CKPT" ]; then
      echo "Evaluating checkpoint: $LATEST"
      lm_eval --model hf \
        --model_args pretrained=$LATEST,dtype=bfloat16 \
        --tasks mmlu,hellaswag \
        --limit 200 \
        --batch_size auto \
        --output_path /checkpoints/${SKYPILOT_TASK_ID}/eval/$(basename $LATEST)/
      LAST_CKPT=$LATEST
    fi
    sleep 60
  done

  wait $TRAIN_PID
```

## Recipe 9: Safety Eval Suite

Full safety evaluation for deployment readiness.

```yaml
name: eval-safety
resources:
  accelerators: A100:1
  disk_size: 200

file_mounts:
  /model:
    source: s3://my-checkpoints/candidate/
  /results:
    name: safety-eval
    store: s3
    mode: MOUNT

setup: |
  pip install lm-eval==0.4.6 vllm==0.6.6

run: |
  lm_eval --model vllm \
    --model_args pretrained=/model,dtype=bfloat16 \
    --tasks truthfulqa_mc2,toxigen,bbq \
    --apply_chat_template \
    --batch_size auto \
    --output_path /results/${SKYPILOT_TASK_ID}/ \
    --log_samples
```

## Tips and Troubleshooting

**OOM during eval:**
- Reduce `--batch_size` to a fixed number (4, 8, 16)
- Use `--model_args ...,max_model_len=2048` to cap sequence length
- Use VLLM backend (better memory management than HF)

**Slow eval:**
- Switch from HF backend to VLLM (2-5x speedup)
- Use `tensor_parallel_size` for multi-GPU
- Use `--limit 500` for faster iteration (not final results)

**Mismatched scores vs leaderboard:**
- Check few-shot count matches exactly
- Verify chat template (applied or not)
- Pin lm-eval version to match leaderboard
- Check if leaderboard uses normalized accuracy (`acc_norm` vs `acc`)

**Results storage pattern:**
```
/results/
  {SKYPILOT_TASK_ID}/
    results.json          # Aggregated scores
    {task_name}/
      samples.jsonl       # Per-sample predictions (if --log_samples)
```
