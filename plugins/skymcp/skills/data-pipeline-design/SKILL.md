---
name: data-pipeline-design
description: Use when designing a data pipeline, curating pretraining data, choosing between NeMo Curator and other tools, configuring tokenization, running deduplication (exact or fuzzy), applying data filtering or quality classifiers, working with FineWeb or Dolma or RedPajama, or assessing data quality for ML training - the production data pipeline design reference
---

# Data Pipeline Design for ML Training

## Standard Pipeline Architecture

A production pretraining data pipeline follows this sequence:

```
Raw Crawl Data
    |
    v
[1] Download & Extract (WARC -> text, PDF -> text, etc.)
    |
    v
[2] Language Detection (fasttext lid.176.bin, ~100 langs)
    |
    v
[3] Heuristic Filters (word count, perplexity, char ratios)
    |
    v
[4] Exact Deduplication (URL-level, hash-based doc-level)
    |
    v
[5] Fuzzy Deduplication (MinHash + LSH, paragraph-level)
    |
    v
[6] Semantic Deduplication (embedding clusters, cosine threshold)
    |
    v
[7] Quality Classifier (fastText model, educational scorer)
    |
    v
[8] PII Redaction (regex + NER for emails, phones, SSNs)
    |
    v
[9] Tokenization (BPE: tiktoken, sentencepiece, HF tokenizer)
    |
    v
Clean Token Stream -> Training
```

Each stage is independently parallelizable. Run stages 3-6 on GPU for large-scale pipelines using NeMo Curator.

## NeMo Curator

GPU-accelerated data curation built on RAPIDS/cuDF. Handles text, image, video, and audio.

### Core Capabilities

| Module | Purpose | Backend |
|--------|---------|---------|
| `nemo_curator.download` | Common Crawl/Wikipedia extraction | CPU (I/O bound) |
| `nemo_curator.filters` | Heuristic quality filters | GPU (cuDF) |
| `nemo_curator.modules.ExactDuplicates` | Hash-based exact dedup | GPU |
| `nemo_curator.modules.FuzzyDuplicates` | MinHash + LSH fuzzy dedup | GPU |
| `nemo_curator.modules.SemanticDuplicates` | Embedding-based semantic dedup | GPU |
| `nemo_curator.classifiers` | Quality/domain/toxicity classifiers | GPU |
| `nemo_curator.pii` | PII detection and redaction | GPU (NER) |

### Quick Start

```python
from nemo_curator import get_client
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules import ExactDuplicates, FuzzyDuplicates
from nemo_curator.filters import WordCountFilter, RepetitiousFilter

# Initialize distributed client (Dask/Ray)
client = get_client(cluster_type="gpu", n_workers=4)

# Load data
dataset = DocumentDataset.read_parquet("/data/raw/*.parquet")

# Heuristic filtering
filtered = WordCountFilter(min_words=50, max_words=100000).filter(dataset)
filtered = RepetitiousFilter(max_ratio=0.3).filter(filtered)

# Exact dedup
exact_dedup = ExactDuplicates(hash_method="md5", id_field="doc_id")
deduped = exact_dedup.compute(filtered)

# Fuzzy dedup
fuzzy_dedup = FuzzyDuplicates(
    seed=42,
    num_hashes=128,
    char_ngrams=5,
    jaccard_threshold=0.8,
)
deduped = fuzzy_dedup.compute(deduped)

# Write clean data
deduped.to_parquet("/data/clean/")
```

### SkyPilot Integration

```yaml
name: data-curation
resources:
  accelerators: A100:4
  disk_size: 2048
  disk_tier: high

file_mounts:
  /data:
    source: s3://my-bucket/crawl-data
    mode: MOUNT_CACHED

setup: |
  pip install nemo-curator[all]

run: |
  python curate.py --input /data/raw --output /data/clean
```

## Quality Filtering Tiers

### Tier 1: Heuristic Filters (Fast, CPU/GPU)

Apply first to remove obvious junk before expensive dedup and classification.

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| Word count | 50 - 100,000 | Remove stubs and dumps |
| Mean word length | 3 - 10 chars | Catch gibberish |
| Symbol-to-word ratio | < 0.1 | Remove symbol-heavy noise |
| URL density | < 0.3 | Remove link farms |
| Bullet/ellipsis lines | < 0.9 ratio | Remove lists-only pages |
| Curly bracket ratio | < 0.1 | Remove code-heavy pages (if not wanted) |
| Stop word presence | > 2 stop words | Catch non-natural language |
| Character n-gram repetition | < 0.3 (top 2-4 gram ratio) | Remove repetitive text |
| Perplexity | < 1000 (KenLM) | Remove incoherent text |
| Terminal punctuation ratio | > 0.1 | Ensure complete sentences |

### Tier 2: Classifier Filters (Medium, GPU)

Train or use pretrained classifiers to score document quality.

**fastText quality classifier** (FineWeb approach):
```python
# Train a quality classifier
# Positive: Wikipedia/textbook content
# Negative: random Common Crawl
import fasttext
model = fasttext.train_supervised(
    input="quality_labels.txt",
    lr=0.1,
    epoch=5,
    wordNgrams=2,
)
# Apply: keep docs where P(high_quality) > 0.5
```

**Educational scoring** (FineWeb-Edu approach):
- Train a classifier on educational content annotations
- Score 0-5 scale: 0 = not educational, 5 = textbook quality
- FineWeb-Edu keeps score >= 3 (yields 1.3T tokens from 15T)

**Toxicity filtering**:
- Use a toxicity classifier (e.g., Jigsaw, HateBERT)
- Remove documents with toxicity score > 0.5
- Or annotate and let the model learn to avoid

### Tier 3: Semantic Filters (Expensive, GPU)

Use embedding models for semantic-level quality:
- Cluster documents by embedding similarity
- Remove outlier clusters (likely spam or near-duplicates)
- Keep diverse representative documents from each cluster

## Deduplication Strategy

### Exact Dedup (Always Run First)

- Hash entire document (MD5/SHA-256)
- Remove byte-identical duplicates
- Also dedup by URL (normalize first: strip params, lowercase)
- Cost: O(N) with hash table

### Fuzzy Dedup (MinHash + LSH)

- Convert documents to character n-gram shingle sets
- Compute MinHash signatures (128-256 hashes)
- Use Locality-Sensitive Hashing to find candidate pairs
- Verify Jaccard similarity > threshold (0.7-0.8)
- Cost: O(N log N) amortized

Key parameters:
- `num_hashes`: 128 (more = more accurate, slower)
- `char_ngrams`: 5 (character-level, not word-level)
- `jaccard_threshold`: 0.8 (lower = more aggressive dedup)
- `num_bands` and `rows_per_band`: control precision/recall tradeoff

### Semantic Dedup

- Embed all documents (e.g., E5-large, BGE-M3)
- Cluster embeddings (k-means or HDBSCAN)
- Within each cluster, remove documents with cosine similarity > 0.95
- Keep the longest or highest-quality document from each near-duplicate group
- Cost: O(N * d) for embedding, O(N^2 / K) for within-cluster comparison

## Available Pretraining Datasets

| Dataset | Tokens | Sources | License | Notes |
|---------|--------|---------|---------|-------|
| FineWeb | 15T | 96 CC snapshots | ODC-By | Best-studied open dataset, strong filters |
| FineWeb-Edu | 1.3T | FineWeb subset | ODC-By | Educational content, score >= 3 |
| FineWeb2 | Varies | Multilingual CC | ODC-By | Multilingual pipeline |
| Dolma | 3T | CC, Wikipedia, books, code | AI2 ImpACT | Fully open (data + code + filters) |
| RedPajama-v2 | 30T | CC | Apache-2.0 | 40+ quality annotations per doc |
| DCLM | 240T raw | CC | MIT | Standardized filtering testbed |
| The Pile | 800B | 22 sources | MIT | Classic, well-studied |
| SlimPajama | 627B | RedPajama-v1 cleaned | Apache-2.0 | Deduped + cleaned |
| StarCoder | 250B | GitHub code | BigCode | 86 programming languages |
| The Stack v2 | 900B+ | GitHub + more | BigCode | Broader code corpus |

### Dataset Selection Guidelines

- **General pretraining**: FineWeb (15T) or Dolma (3T) as base, mix with code (StarCoder/Stack)
- **Domain-specific**: Start with general corpus, add domain data at 10-30% mix ratio
- **Small-scale research**: FineWeb-Edu (1.3T) gives best quality-per-token
- **Multilingual**: FineWeb2 or CulturaX
- **Reproducibility**: Dolma (fully open pipeline) or DCLM (standardized)

## Tokenization

### Tokenizer Selection

| Tokenizer | Vocabulary | Used By | Speed |
|-----------|-----------|---------|-------|
| tiktoken (cl100k) | 100,256 | GPT-4, Claude | Fastest (Rust) |
| tiktoken (o200k) | 200,000 | GPT-4o | Fastest (Rust) |
| SentencePiece (BPE) | 32,000-128,000 | Llama, Gemma | Fast (C++) |
| HF Tokenizers | Configurable | Most HF models | Fast (Rust) |

### Training a Custom Tokenizer

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=32768,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|padding|>", "<|begin_of_text|>"],
)
tokenizer.train(files=["corpus_sample.txt"], trainer=trainer)
tokenizer.save("tokenizer.json")
```

### Tokenization at Scale

For terabyte-scale tokenization, shard the work across nodes:

```yaml
# tokenize.yaml (SkyPilot)
num_nodes: 8
resources:
  cpus: 96

run: |
  python tokenize_shard.py \
    --input /data/clean/shard_${SKYPILOT_NODE_RANK}/ \
    --output /data/tokens/shard_${SKYPILOT_NODE_RANK}/ \
    --tokenizer tokenizer.json
```

## Data Mixing

For multi-source training, mix datasets by domain:

| Domain | Typical Mix | Purpose |
|--------|------------|---------|
| Web text | 60-70% | General knowledge, language modeling |
| Code | 10-20% | Reasoning, structured output |
| Books / papers | 5-10% | Long-form coherence, depth |
| Conversations | 5-10% | Dialogue, instruction following |
| Math / science | 2-5% | Quantitative reasoning |

Use epoch-based or token-based sampling to maintain ratios during training.

See [references/nemo-curator-guide.md](references/nemo-curator-guide.md) for complete NeMo Curator configuration.
See [references/dataset-comparison.md](references/dataset-comparison.md) for detailed dataset analysis.
