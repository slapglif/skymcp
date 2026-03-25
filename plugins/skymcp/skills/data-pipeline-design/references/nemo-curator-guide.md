# NeMo Curator Complete Guide

GPU-accelerated data curation for pretraining datasets. Built on RAPIDS/cuDF with Dask and Ray distributed backends.

## Installation

```bash
# Full installation (GPU)
pip install nemo-curator[all]

# CPU-only (for development/testing)
pip install nemo-curator

# With specific backend
pip install nemo-curator[ray]    # Ray backend
pip install nemo-curator[dask]   # Dask backend (default)
```

## Architecture

NeMo Curator operates on `DocumentDataset` objects, which wrap Dask DataFrames. Each row is a document with at minimum a `text` column and an `id` column.

```
DocumentDataset (Dask DataFrame)
├── text: str          # Document text
├── id: str            # Unique identifier
├── language: str      # Detected language (after LID)
├── quality_score: float  # Quality classifier output
└── ... (custom columns)
```

## Distributed Client Setup

### Dask (Default)

```python
from nemo_curator import get_client

# Local GPU cluster
client = get_client(cluster_type="gpu", n_workers=4)

# Local CPU cluster
client = get_client(cluster_type="cpu", n_workers=16)

# Remote Dask scheduler
from dask.distributed import Client
client = Client("tcp://scheduler:8786")
```

### Ray

```python
import ray
ray.init(address="auto")

from nemo_curator import get_client
client = get_client(cluster_type="ray")
```

## Data Loading

### From Parquet

```python
from nemo_curator.datasets import DocumentDataset

dataset = DocumentDataset.read_parquet(
    "/data/raw/*.parquet",
    columns=["text", "url", "timestamp"],
)
```

### From JSONL

```python
dataset = DocumentDataset.read_json(
    "/data/raw/*.jsonl",
    lines=True,
)
```

### From Common Crawl

```python
from nemo_curator.download import download_common_crawl

dataset = download_common_crawl(
    output_path="/data/cc/",
    start_snapshot="CC-MAIN-2024-10",
    end_snapshot="CC-MAIN-2024-22",
    output_type="parquet",
)
```

### From Wikipedia

```python
from nemo_curator.download import download_wikipedia

dataset = download_wikipedia(
    output_path="/data/wiki/",
    language="en",
    dump_date="20240301",
)
```

## Language Detection

```python
from nemo_curator.modules import AddId
from nemo_curator.filters import FastTextLangId

# Add unique IDs
add_id = AddId(id_field="doc_id", id_prefix="doc")
dataset = add_id(dataset)

# Detect language
lid = FastTextLangId(model_path="/models/lid.176.bin")
dataset = lid(dataset)

# Filter to English
dataset = dataset.filter(lambda row: row["language"] == "en")
```

## Heuristic Filters

### Built-in Filters

```python
from nemo_curator.filters import (
    WordCountFilter,
    WordsPerLineFilter,
    MeanWordLengthFilter,
    SymbolToWordRatio,
    BulletsRatio,
    EllipsisRatio,
    AlphaFilter,
    URLFilter,
    RepetitiousFilter,
    RepeatedLinesFilter,
    RepeatedParagraphsFilter,
    PunctuationFilter,
    BoilerplateStringFilter,
)

# Chain filters
from nemo_curator.modules import ScoreFilter

filters = [
    WordCountFilter(min_words=50, max_words=100000),
    MeanWordLengthFilter(min_mean=3, max_mean=10),
    SymbolToWordRatio(max_ratio=0.1),
    BulletsRatio(max_ratio=0.9),
    EllipsisRatio(max_ratio=0.3),
    AlphaFilter(min_ratio=0.7),
    URLFilter(max_ratio=0.3),
    RepetitiousFilter(
        top_n_grams=[2, 3, 4],
        max_ratio=[0.2, 0.18, 0.16],
    ),
    RepeatedLinesFilter(max_ratio=0.3),
    RepeatedParagraphsFilter(max_ratio=0.3),
    PunctuationFilter(min_ratio=0.1),
]

for f in filters:
    dataset = ScoreFilter(f).filter(dataset)
```

### Custom Filter

```python
from nemo_curator.filters import DocumentFilter

class CustomPerplexityFilter(DocumentFilter):
    def __init__(self, max_perplexity=1000):
        super().__init__()
        self.max_perplexity = max_perplexity

    def score_document(self, text: str) -> float:
        """Return a score. Higher = worse quality."""
        # Your perplexity computation here
        return compute_perplexity(text)

    def keep_document(self, score: float) -> bool:
        return score < self.max_perplexity

custom_filter = CustomPerplexityFilter(max_perplexity=500)
dataset = ScoreFilter(custom_filter).filter(dataset)
```

## Deduplication

### Exact Deduplication

```python
from nemo_curator.modules import ExactDuplicates

exact_dedup = ExactDuplicates(
    id_field="doc_id",
    text_field="text",
    hash_method="md5",         # md5, sha256
    cache_dir="/tmp/dedup_cache",
)

# Returns dataset with duplicates marked
result = exact_dedup(dataset)

# Remove duplicates
deduped = result.filter(lambda row: not row["is_duplicate"])
```

### Fuzzy Deduplication (MinHash + LSH)

```python
from nemo_curator.modules import FuzzyDuplicates, FuzzyDuplicatesConfig

config = FuzzyDuplicatesConfig(
    seed=42,
    char_ngrams=5,              # Character n-gram size for shingling
    num_hashes=128,             # Number of MinHash signatures
    num_buckets=64,             # LSH buckets (more = higher precision)
    buckets_per_shuffle=8,      # Memory-speed tradeoff
    jaccard_threshold=0.8,      # Similarity threshold
    false_positive_weight=0.5,  # Weight for false positive penalty
    false_negative_weight=0.5,  # Weight for false negative penalty
)

fuzzy_dedup = FuzzyDuplicates(
    config=config,
    id_field="doc_id",
    text_field="text",
    cache_dir="/tmp/fuzzy_cache",
)

result = fuzzy_dedup(dataset)
deduped = result.filter(lambda row: not row["is_fuzzy_duplicate"])
```

### Semantic Deduplication

```python
from nemo_curator.modules import SemanticDuplicates, SemanticDuplicatesConfig

config = SemanticDuplicatesConfig(
    model_name="intfloat/e5-large-v2",
    embedding_batch_size=128,
    max_memory_usage="40GB",
    cosine_threshold=0.95,       # Very high = only near-identical
    num_clusters=1000,           # K-means clusters
    which_to_keep="longest",     # longest, shortest, first
)

sem_dedup = SemanticDuplicates(
    config=config,
    id_field="doc_id",
    text_field="text",
    cache_dir="/tmp/sem_cache",
)

result = sem_dedup(dataset)
```

## Quality Classifiers

### Built-in Quality Classifier

```python
from nemo_curator.classifiers import QualityClassifier

classifier = QualityClassifier(
    model_path="/models/quality_classifier",
    batch_size=256,
    device="cuda",
)

# Adds quality_score column
dataset = classifier(dataset)

# Keep high-quality documents
dataset = dataset.filter(lambda row: row["quality_score"] > 0.5)
```

### Domain Classifier

```python
from nemo_curator.classifiers import DomainClassifier

domain_clf = DomainClassifier(
    model_path="/models/domain_classifier",
    batch_size=256,
)

dataset = domain_clf(dataset)
# Adds domain_pred column (e.g., "science", "news", "social_media")
```

### Educational Content Scoring

```python
from nemo_curator.classifiers import EducationalClassifier

edu_clf = EducationalClassifier(
    model_path="/models/edu_classifier",
    batch_size=256,
)

dataset = edu_clf(dataset)
# Adds edu_score column (0-5 scale)
# FineWeb-Edu uses threshold >= 3
dataset = dataset.filter(lambda row: row["edu_score"] >= 3)
```

### Toxicity Classifier

```python
from nemo_curator.classifiers import ToxicityClassifier

tox_clf = ToxicityClassifier(
    model_path="unitary/toxic-bert",
    batch_size=256,
    threshold=0.5,
)

dataset = tox_clf(dataset)
dataset = dataset.filter(lambda row: row["toxicity_score"] < 0.5)
```

## PII Redaction

```python
from nemo_curator.pii import PiiRedactor, PiiConfig

config = PiiConfig(
    supported_entities=[
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "IP_ADDRESS",
        "PERSON",
    ],
    anonymize_action="replace",    # replace, hash, mask
    language="en",
)

redactor = PiiRedactor(config=config)
dataset = redactor(dataset)
```

## Complete Pipeline Example

```python
from nemo_curator import get_client
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules import (
    AddId, ExactDuplicates, FuzzyDuplicates,
    FuzzyDuplicatesConfig, ScoreFilter,
)
from nemo_curator.filters import (
    WordCountFilter, MeanWordLengthFilter,
    RepetitiousFilter, URLFilter,
)
from nemo_curator.classifiers import QualityClassifier
from nemo_curator.pii import PiiRedactor

# Initialize
client = get_client(cluster_type="gpu", n_workers=8)

# Load
dataset = DocumentDataset.read_parquet("/data/raw/*.parquet")

# Add IDs
dataset = AddId(id_field="doc_id")(dataset)

# Heuristic filters
for f in [
    WordCountFilter(min_words=50, max_words=100000),
    MeanWordLengthFilter(min_mean=3, max_mean=10),
    URLFilter(max_ratio=0.3),
    RepetitiousFilter(top_n_grams=[2, 3, 4], max_ratio=[0.2, 0.18, 0.16]),
]:
    dataset = ScoreFilter(f).filter(dataset)

# Exact dedup
dataset = ExactDuplicates(id_field="doc_id", hash_method="md5")(dataset)
dataset = dataset.filter(lambda row: not row["is_duplicate"])

# Fuzzy dedup
fuzzy_config = FuzzyDuplicatesConfig(
    num_hashes=128, char_ngrams=5, jaccard_threshold=0.8,
)
dataset = FuzzyDuplicates(config=fuzzy_config, id_field="doc_id")(dataset)
dataset = dataset.filter(lambda row: not row["is_fuzzy_duplicate"])

# Quality classifier
dataset = QualityClassifier(model_path="/models/quality")(dataset)
dataset = dataset.filter(lambda row: row["quality_score"] > 0.5)

# PII redaction
dataset = PiiRedactor()(dataset)

# Save
dataset.to_parquet("/data/clean/")

print(f"Pipeline complete. Output: /data/clean/")
```

## SkyPilot Deployment

```yaml
name: data-pipeline
resources:
  accelerators: A100:8
  disk_size: 4096
  disk_tier: high

file_mounts:
  /data:
    source: s3://my-bucket/crawl-data
    mode: MOUNT_CACHED
  /models:
    source: s3://my-bucket/classifier-models
    mode: COPY

setup: |
  pip install nemo-curator[all]

run: |
  python pipeline.py \
    --input /data/raw \
    --output /data/clean \
    --workers 8
```

## Performance Tips

1. **Use GPU workers** for filtering and dedup -- 10-50x faster than CPU.
2. **Shard data into Parquet files** of 100-500 MB each for optimal parallel processing.
3. **Run dedup stages sequentially** -- exact first (fast), then fuzzy (medium), then semantic (slow).
4. **Cache intermediate results** to disk between stages so you can restart without reprocessing.
5. **Monitor memory** -- fuzzy dedup with MinHash can use significant RAM. Reduce `num_hashes` or `buckets_per_shuffle` if OOM.
6. **Use MOUNT_CACHED** in SkyPilot for input data -- avoids downloading everything upfront.
