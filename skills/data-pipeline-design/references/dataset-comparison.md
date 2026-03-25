# Pretraining Dataset Comparison

Detailed comparison of available open pretraining datasets for LLM training.

## Dataset Overview

### FineWeb (HuggingFace)

- **Size**: 15 trillion tokens
- **Sources**: 96 Common Crawl snapshots (2013-2024)
- **License**: ODC-By 1.0
- **Format**: Parquet (sharded)
- **HuggingFace**: `HuggingFaceFW/fineweb`

**Pipeline**:
1. URL dedup (normalize + hash)
2. trafilatura text extraction
3. Language detection (fastText, keep English > 0.65)
4. Heuristic filters (FineWeb-specific set)
5. MinHash fuzzy dedup (n=5 char-grams, 128 hashes, 0.7 threshold)
6. C4 filter refinements

**Strengths**:
- Best-studied open dataset with published ablations
- Proven to match or beat other open datasets in downstream benchmarks
- Extensive documentation of filtering decisions

**Weaknesses**:
- English only (see FineWeb2 for multilingual)
- 15T tokens may be more than needed for small-scale research

**Best for**: General English pretraining at any scale.

### FineWeb-Edu (HuggingFace)

- **Size**: 1.3 trillion tokens
- **Sources**: FineWeb subset
- **License**: ODC-By 1.0
- **HuggingFace**: `HuggingFaceFW/fineweb-edu`

**Pipeline**: FineWeb pipeline + educational content classifier (score >= 3 out of 5).

**Strengths**:
- Highest quality-per-token of any open dataset
- Models trained on FineWeb-Edu outperform those trained on 10x larger datasets
- Excellent for knowledge-intensive benchmarks (MMLU, ARC, HellaSwag)

**Weaknesses**:
- Biased toward educational/formal content
- May underperform on conversational or creative tasks

**Best for**: Knowledge-intensive models, small-to-medium scale pretraining (1B-7B), research.

### FineWeb2 (HuggingFace)

- **Size**: Varies by language
- **Sources**: Common Crawl, multilingual
- **License**: ODC-By 1.0

**Pipeline**: FineWeb methodology applied to 100+ languages with per-language quality filtering.

**Best for**: Multilingual pretraining.

### Dolma (AI2)

- **Size**: 3 trillion tokens
- **Sources**: Common Crawl, Wikipedia, Project Gutenberg, peS2o (academic), The Stack (code), C4 (re-processed)
- **License**: AI2 ImpACT Low-Risk
- **GitHub**: `allenai/dolma`

**Pipeline**:
1. Multi-source extraction
2. Per-source filtering (custom per domain)
3. URL and document-level dedup
4. Paragraph-level fuzzy dedup (Bloom filter)
5. Quality classification
6. PII masking

**Strengths**:
- Fully open: data, code, and documentation
- Multi-source diversity (web + books + academic + code)
- Reproducible pipeline

**Weaknesses**:
- Smaller than FineWeb/RedPajama
- Some sources have restrictive upstream licenses (check per-subset)

**Best for**: Reproducible research, multi-domain pretraining.

### RedPajama-v2 (Together AI)

- **Size**: 30 trillion raw tokens (before filtering)
- **Sources**: 84 Common Crawl snapshots
- **License**: Apache 2.0
- **HuggingFace**: `togethercomputer/RedPajama-Data-V2`

**Features**:
- 40+ quality annotations per document (not just binary keep/drop)
- Annotations include: perplexity, quality classifier scores, dedup flags, toxicity, language
- Users choose their own filtering thresholds

**Strengths**:
- Largest open dataset by raw token count
- Pre-computed quality annotations save significant compute
- Flexible: apply your own filtering criteria

**Weaknesses**:
- Requires filtering -- raw data includes low-quality content
- No "default" high-quality subset (you must define thresholds)
- Common Crawl only (no books, academic, or code diversity)

**Best for**: Teams who want to experiment with data filtering strategies.

### DCLM (DataComp for Language Models)

- **Size**: 240 trillion raw tokens
- **Sources**: Common Crawl
- **License**: MIT (code), data license varies
- **GitHub**: `mlfoundations/dclm`

**Features**:
- Standardized filtering testbed (like ImageNet for data curation)
- Baseline filters provided, researchers submit improved filters
- Leaderboard for data quality competitions

**Strengths**:
- Standardized evaluation framework
- Encourages reproducible data curation research
- Massive scale for experimentation

**Weaknesses**:
- Research-oriented, not production-ready
- Requires significant compute to process

**Best for**: Data curation research, filtering methodology development.

### The Pile (EleutherAI)

- **Size**: 825 billion tokens
- **Sources**: 22 curated sources including PubMed, ArXiv, GitHub, StackExchange, Wikipedia, USPTO, FreeLaw, etc.
- **License**: MIT (code), mixed (data)

**Strengths**:
- Well-studied, diverse source mix
- Curated per-source quality
- Strong baseline for research

**Weaknesses**:
- Smaller than modern datasets
- Some sources have licensing concerns
- Last updated 2022

**Best for**: Baseline comparison, small-scale research.

### SlimPajama (CerebrasAI)

- **Size**: 627 billion tokens
- **Sources**: RedPajama-v1 (cleaned and deduped)
- **License**: Apache 2.0
- **HuggingFace**: `cerebras/SlimPajama-627B`

**Pipeline**: RedPajama-v1 with aggressive dedup (MinHash + exact) and quality filtering.

**Strengths**:
- Clean, deduplicated subset of RedPajama
- Smaller = cheaper to train on
- Well-documented filtering decisions

**Weaknesses**:
- Based on older RedPajama-v1 pipeline

**Best for**: Budget-conscious pretraining, quick experiments.

### Code Datasets

| Dataset | Tokens | Languages | License |
|---------|--------|-----------|---------|
| StarCoder | 250B | 86 programming languages | BigCode Open RAIL-M |
| The Stack v2 | 900B+ | 600+ languages | BigCode Open RAIL-M |
| CodeParrot | 180B | Python only | Apache 2.0 |

## Quality vs Quantity Tradeoffs

| Strategy | Tokens | Quality | Benchmark Performance |
|----------|--------|---------|----------------------|
| FineWeb (all) | 15T | Medium-High | Good |
| FineWeb-Edu | 1.3T | Very High | Better (per token) |
| Custom filtered (FineWeb + classifier) | 2-5T | High | Best |
| RedPajama-v2 (lightly filtered) | 15-20T | Medium | Good at scale |
| Multi-source (Dolma approach) | 3T | High (diverse) | Good |

**Key insight**: A smaller, higher-quality dataset consistently outperforms a larger, lower-quality one up to approximately 10x size difference. Beyond 10x, raw scale wins.

## Data Mixing Ratios

Based on published results from Llama, Gemma, and other open models:

| Domain | Typical Ratio | Purpose |
|--------|--------------|---------|
| Web crawl (filtered) | 60-70% | General knowledge, language |
| Code | 10-20% | Reasoning, structured output |
| Academic/scientific | 5-10% | Technical knowledge, precision |
| Books | 3-5% | Long-form coherence |
| Wikipedia | 2-5% | Factual knowledge |
| Conversations/forums | 2-5% | Dialogue, informal language |
| Math | 1-3% | Quantitative reasoning |

### Epoch Strategy

For tokens T_domain in domain D and total training tokens T_total:

- If T_domain > T_total * ratio_D: Sample without replacement (single epoch)
- If T_domain < T_total * ratio_D: Repeat with 2-5 epochs maximum

Repeating data beyond 4-5 epochs shows diminishing returns and risk of memorization.

## Practical Recommendations

### For Research (1B-7B models)

```
Base: FineWeb-Edu (1.3T tokens)
Code: StarCoder (50B subset)
Total: ~1.35T tokens
Training: 300B-1T tokens (single epoch of subset)
```

### For Production (7B-70B models)

```
Base: FineWeb (5T subset, quality-filtered)
Code: The Stack v2 (500B subset)
Academic: peS2o from Dolma (100B)
Books: Project Gutenberg from Dolma (5B)
Wikipedia: 10B
Total: ~5.6T tokens
Training: 2-5T tokens
```

### For Maximum Scale (70B+ models)

```
Base: FineWeb (15T) + RedPajama-v2 (custom-filtered 10T)
Code: The Stack v2 (900B)
Academic: peS2o (200B) + arXiv (100B)
Books: 50B
Wikipedia: 20B (multi-epoch)
Math: 30B (curated)
Total: ~26T tokens
Training: 5-15T tokens
```

## Data Downloading

### FineWeb via HuggingFace

```python
from datasets import load_dataset

# Stream (no full download)
dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

# Download specific snapshot
dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="CC-MAIN-2024-10",
    split="train",
)

# Save to disk
dataset.save_to_disk("/data/fineweb/CC-MAIN-2024-10")
```

### Dolma via CLI

```bash
pip install dolma
dolma download --dataset dolma-v1_7 --output /data/dolma/ --subset cc_en_head
```

### RedPajama-v2

```python
from datasets import load_dataset

dataset = load_dataset(
    "togethercomputer/RedPajama-Data-V2",
    name="default",
    partition="head_middle",  # head_middle, tail
    snapshots=["2024-10"],
    languages=["en"],
    streaming=True,
)
```
