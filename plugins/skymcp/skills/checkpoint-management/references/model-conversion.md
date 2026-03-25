# Model Format Conversion Recipes

Step-by-step recipes for converting between model formats. Every command is copy-pasteable.

## Format Overview

```
                  Training                    Deployment
                  --------                    ----------
SafeTensors <---> PyTorch (.bin)              GGUF (llama.cpp/Ollama)
    |                                           ^
    |                                           |
    +-- HuggingFace Hub format                  +-- quantized (Q4, Q5, Q8, etc.)
    |                                           |
    +-- merge LoRA here                         +-- imatrix for quality
    |
    +---> ONNX ---> TensorRT / OpenVINO / DirectML
```

## Recipe 1: PyTorch .bin to SafeTensors

Convert legacy PyTorch pickle format to SafeTensors.

```python
from safetensors.torch import save_file
import torch
import json
import os

def convert_bin_to_safetensors(model_dir):
    """Convert all .bin files in a HF model directory to .safetensors."""
    # Find all .bin shard files
    bin_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".bin"))

    for bin_file in bin_files:
        print(f"Converting {bin_file}...")
        state_dict = torch.load(
            os.path.join(model_dir, bin_file),
            map_location="cpu",
            weights_only=True
        )

        # Create corresponding .safetensors file
        st_file = bin_file.replace(".bin", ".safetensors")
        save_file(state_dict, os.path.join(model_dir, st_file))

    # Update model index if sharded
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    old_index = os.path.join(model_dir, "pytorch_model.bin.index.json")
    if os.path.exists(old_index):
        with open(old_index) as f:
            index = json.load(f)
        # Rename .bin references to .safetensors
        index["weight_map"] = {
            k: v.replace(".bin", ".safetensors")
            for k, v in index["weight_map"].items()
        }
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)

    print("Done. You can now delete the .bin files.")

# Usage
convert_bin_to_safetensors("/path/to/model")
```

**Or use the HuggingFace CLI:**
```bash
# Download and auto-convert
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir /model
```

## Recipe 2: SafeTensors/HF to GGUF

Convert a HuggingFace model to GGUF format for llama.cpp and Ollama.

### Step 1: Get llama.cpp conversion tools

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements/requirements-convert_hf_to_gguf.txt
```

### Step 2: Convert to GGUF FP16

```bash
python convert_hf_to_gguf.py /path/to/hf-model \
  --outfile /output/model-f16.gguf \
  --outtype f16
```

**Supported output types:**
- `f32` -- full precision (huge, rarely needed)
- `f16` -- half precision (standard conversion target)
- `bf16` -- bfloat16 (if model was trained in bf16)
- `q8_0` -- 8-bit quantized during conversion (saves a step)

### Step 3: Generate Importance Matrix (Recommended)

Importance matrix (imatrix) tracks which weights matter most, improving quantization quality at low bit widths.

```bash
# Build llama.cpp if not already built
cmake -B build && cmake --build build --config Release

# Generate imatrix from calibration data
./build/bin/llama-imatrix \
  -m /output/model-f16.gguf \
  -f /path/to/calibration.txt \
  -o /output/imatrix.dat \
  --chunks 100

# calibration.txt: representative text samples, ~100-500 lines
# Good sources: wikitext, your target domain text
```

### Step 4: Quantize

```bash
# Without imatrix
./build/bin/llama-quantize \
  /output/model-f16.gguf \
  /output/model-Q4_K_M.gguf \
  Q4_K_M

# With imatrix (recommended for Q4 and below)
./build/bin/llama-quantize \
  --imatrix /output/imatrix.dat \
  /output/model-f16.gguf \
  /output/model-Q4_K_M.gguf \
  Q4_K_M
```

### Common Quantization Types

| Type | Bits/Weight | 7B Size | 13B Size | 70B Size | Notes |
|------|-------------|---------|----------|----------|-------|
| Q2_K | ~2.6 | 2.8 GB | 5.2 GB | 27 GB | Very lossy, last resort |
| Q3_K_S | ~3.4 | 3.0 GB | 5.5 GB | 29 GB | Small variant |
| Q3_K_M | ~3.9 | 3.3 GB | 6.0 GB | 32 GB | Medium variant |
| Q4_0 | ~4.5 | 3.8 GB | 7.0 GB | 37 GB | Legacy, use Q4_K_M instead |
| Q4_K_S | ~4.6 | 4.0 GB | 7.4 GB | 39 GB | Small variant |
| **Q4_K_M** | **~4.8** | **4.4 GB** | **8.0 GB** | **42 GB** | **Default recommendation** |
| Q5_K_S | ~5.5 | 4.8 GB | 8.7 GB | 46 GB | Small variant |
| Q5_K_M | ~5.7 | 5.1 GB | 9.3 GB | 49 GB | Good quality/size balance |
| Q6_K | ~6.6 | 5.9 GB | 10.7 GB | 56 GB | Near-lossless |
| Q8_0 | ~8.5 | 7.7 GB | 14.0 GB | 73 GB | Minimal loss |
| F16 | 16 | 14.0 GB | 26.0 GB | 137 GB | No quantization |

### Step 5: Verify

```bash
# Quick generation test
./build/bin/llama-cli \
  -m /output/model-Q4_K_M.gguf \
  -p "The meaning of life is" \
  -n 50

# Perplexity test (wikitext)
./build/bin/llama-perplexity \
  -m /output/model-Q4_K_M.gguf \
  -f /path/to/wikitext-2-raw/wiki.test.raw
```

## Recipe 3: Merge LoRA + Convert to GGUF

Complete pipeline from adapter to deployable GGUF.

```bash
#!/bin/bash
set -euo pipefail

BASE_MODEL="meta-llama/Llama-3.1-8B"
ADAPTER_PATH="/checkpoints/my-lora-adapter"
OUTPUT_DIR="/output"

# Step 1: Merge adapter into base
python3 -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('Loading base model...')
base = AutoModelForCausalLM.from_pretrained(
    '${BASE_MODEL}',
    torch_dtype=torch.bfloat16,
    device_map='cpu'
)

print('Loading adapter...')
model = PeftModel.from_pretrained(base, '${ADAPTER_PATH}')

print('Merging...')
merged = model.merge_and_unload()

print('Saving merged model...')
merged.save_pretrained('${OUTPUT_DIR}/merged', safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained('${BASE_MODEL}')
tokenizer.save_pretrained('${OUTPUT_DIR}/merged')
print('Done merging.')
"

# Step 2: Convert to GGUF
python3 llama.cpp/convert_hf_to_gguf.py \
  ${OUTPUT_DIR}/merged \
  --outfile ${OUTPUT_DIR}/model-f16.gguf \
  --outtype f16

# Step 3: Generate imatrix (optional but recommended)
./llama.cpp/build/bin/llama-imatrix \
  -m ${OUTPUT_DIR}/model-f16.gguf \
  -f calibration.txt \
  -o ${OUTPUT_DIR}/imatrix.dat \
  --chunks 100

# Step 4: Quantize
for QUANT in Q4_K_M Q5_K_M Q8_0; do
  echo "Quantizing ${QUANT}..."
  ./llama.cpp/build/bin/llama-quantize \
    --imatrix ${OUTPUT_DIR}/imatrix.dat \
    ${OUTPUT_DIR}/model-f16.gguf \
    ${OUTPUT_DIR}/model-${QUANT}.gguf \
    ${QUANT}
done

echo "All quantizations complete:"
ls -lh ${OUTPUT_DIR}/model-*.gguf
```

## Recipe 4: HuggingFace to ONNX

For cross-framework deployment (TensorRT, OpenVINO, DirectML).

```bash
pip install optimum[exporters]

# Export to ONNX
optimum-cli export onnx \
  --model /path/to/hf-model \
  --task text-generation \
  /output/onnx-model/

# Validate export
optimum-cli validate onnx /output/onnx-model/

# Quantize ONNX (dynamic int8)
optimum-cli onnxruntime quantize \
  --onnx_model /output/onnx-model/ \
  --output /output/onnx-model-int8/ \
  --per_channel
```

## Recipe 5: ONNX to TensorRT

For NVIDIA GPU deployment with maximum inference speed.

```bash
# Using trtllm-build (TensorRT-LLM)
pip install tensorrt_llm

# Convert HF model to TensorRT-LLM format
python3 -c "
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model='/path/to/hf-model')
# TensorRT-LLM handles conversion internally
"

# Or manual: HF -> ONNX -> TensorRT
trtexec \
  --onnx=/output/onnx-model/model.onnx \
  --saveEngine=/output/model.trt \
  --fp16 \
  --workspace=8192
```

## Recipe 6: SafeTensors to Ollama (via GGUF)

Deploy a custom model to Ollama for local serving.

```bash
# 1. Convert and quantize (see Recipe 2 above)
# Result: model-Q4_K_M.gguf

# 2. Create Modelfile
cat > Modelfile << 'EOF'
FROM ./model-Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|eot_id|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# 3. Create Ollama model
ollama create my-model -f Modelfile

# 4. Test
ollama run my-model "Hello, world!"
```

## Recipe 7: AWQ Quantization (GPU-Optimized)

AWQ (Activation-Aware Weight Quantization) is faster than GGUF on GPU.

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "/path/to/hf-model"
quant_path = "/output/model-awq"

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Quantize
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# Use with vLLM
# vllm serve /output/model-awq --quantization awq
```

## Recipe 8: GPTQ Quantization

Alternative GPU quantization. Well-supported in HuggingFace and vLLM.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_path = "meta-llama/Llama-3.1-8B"

# Configure quantization
quantization_config = GPTQConfig(
    bits=4,
    dataset="c4",
    tokenizer=AutoTokenizer.from_pretrained(model_path)
)

# Load and quantize
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)

# Save
model.save_pretrained("/output/model-gptq-4bit")
```

## Conversion Compatibility Matrix

| From \ To | SafeTensors | GGUF | ONNX | AWQ | GPTQ |
|-----------|-------------|------|------|-----|------|
| SafeTensors | -- | convert_hf_to_gguf | optimum export | autoawq | transformers |
| PyTorch .bin | save_file() | convert first | optimum export | convert first | convert first |
| GGUF | N/A (lossy) | -- | N/A | N/A | N/A |
| ONNX | N/A | N/A | -- | N/A | N/A |

**Important:** GGUF and other quantized formats are one-way. Always keep the original FP16/BF16 checkpoint.

## SkyPilot Conversion Job

Run format conversion on a cloud GPU (needed for AWQ/GPTQ which require GPU).

```yaml
name: model-convert
resources:
  accelerators: A100:1
  disk_size: 500

file_mounts:
  /model:
    source: s3://my-checkpoints/merged-model/
  /output:
    name: converted-models
    store: s3
    mode: MOUNT

setup: |
  pip install autoawq vllm transformers accelerate
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp && cmake -B build && cmake --build build --config Release
  pip install -r requirements/requirements-convert_hf_to_gguf.txt

run: |
  # GGUF conversion + quantization
  python llama.cpp/convert_hf_to_gguf.py /model \
    --outfile /output/model-f16.gguf --outtype f16

  ./llama.cpp/build/bin/llama-imatrix \
    -m /output/model-f16.gguf \
    -f /path/to/calibration.txt \
    -o /output/imatrix.dat --chunks 100

  for Q in Q4_K_M Q5_K_M Q8_0; do
    ./llama.cpp/build/bin/llama-quantize \
      --imatrix /output/imatrix.dat \
      /output/model-f16.gguf \
      /output/model-${Q}.gguf ${Q}
  done

  # AWQ quantization
  python3 -c "
  from awq import AutoAWQForCausalLM
  from transformers import AutoTokenizer
  model = AutoAWQForCausalLM.from_pretrained('/model')
  tokenizer = AutoTokenizer.from_pretrained('/model')
  model.quantize(tokenizer, quant_config={'zero_point':True,'q_group_size':128,'w_bit':4,'version':'GEMM'})
  model.save_quantized('/output/model-awq')
  tokenizer.save_pretrained('/output/model-awq')
  "

  echo "All conversions complete:"
  ls -lh /output/
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `convert_hf_to_gguf.py` fails with "unknown model" | Update llama.cpp. New architectures added frequently. |
| GGUF quantization produces garbage | Ensure FP16 conversion was clean. Test FP16 GGUF first. |
| AWQ OOM during quantization | Use `device_map="auto"` or reduce calibration samples. |
| Merged LoRA model produces different results | Check that base model version matches exactly. BF16 vs FP16 matters. |
| ONNX export fails for custom architectures | Use `--opset` flag to try different ONNX opset versions. |
