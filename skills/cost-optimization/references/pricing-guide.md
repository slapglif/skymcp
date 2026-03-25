# GPU Pricing Guide Across Clouds

Detailed GPU pricing reference for cost estimation and cloud selection. Prices are approximate as of early 2026 and fluctuate. Always verify with `sky gpus --all` for current live pricing.

## NVIDIA H100 80GB (SXM5)

The fastest GPU for training. Preferred for large-scale training and throughput-critical workloads.

| Cloud | Instance Type | Spot ($/hr) | On-Demand ($/hr) | Notes |
|-------|---------------|-------------|-------------------|-------|
| AWS | p5.48xlarge (8x) | $18-25/8=$2.25-3.12 per GPU | $80-98/8=$10-12.25 per GPU | EFA networking available |
| GCP | a3-highgpu-8g (8x) | $16-22/8=$2.00-2.75 per GPU | $72-88/8=$9-11 per GPU | GPUDirect-TCPX |
| Lambda | gpu_8x_h100_sxm5 | N/A (no spot) | $2.49-3.29 per GPU | Often cheapest on-demand |
| Azure | ND H100 v5 (8x) | $18-24/8=$2.25-3.00 per GPU | $80-96/8=$10-12 per GPU | InfiniBand |
| RunPod | H100 SXM | $2.39 per GPU | $3.89 per GPU | Community cloud |
| CoreWeave | H100 SXM | N/A | $2.23-2.83 per GPU | Reserved pricing available |

**Best deal:** Lambda on-demand ($2.49-3.29/GPU/hr) or RunPod spot ($2.39/GPU/hr)
**Best availability:** AWS (most regions, best spot recovery)

## NVIDIA A100 80GB (SXM4)

Workhorse GPU. Best price/performance for most training and inference workloads.

| Cloud | Instance Type | Spot ($/hr) | On-Demand ($/hr) | Notes |
|-------|---------------|-------------|-------------------|-------|
| AWS | p4de.24xlarge (8x) | $8-14/8=$1.00-1.75 per GPU | $40-48/8=$5.00-6.00 per GPU | Older but widely available |
| GCP | a2-ultragpu-8g (8x) | $8-12/8=$1.00-1.50 per GPU | $36-44/8=$4.50-5.50 per GPU | Good spot availability |
| Lambda | gpu_8x_a100_80gb | N/A | $1.29-1.79 per GPU | Very competitive on-demand |
| Azure | ND A100 v4 (8x) | $8-14/8=$1.00-1.75 per GPU | $38-48/8=$4.75-6.00 per GPU | InfiniBand |
| RunPod | A100 80GB SXM | $1.19 per GPU | $1.89 per GPU | Community cloud |

**Best deal:** Lambda ($1.29-1.79/GPU/hr on-demand) or RunPod ($1.19/GPU/hr)

## NVIDIA A100 40GB (PCIe / SXM)

Budget A100 option. Good for models up to ~30B parameters or inference.

| Cloud | Instance Type | Spot ($/hr) | On-Demand ($/hr) | Notes |
|-------|---------------|-------------|-------------------|-------|
| AWS | p4d.24xlarge (8x) | $7-11/8=$0.88-1.38 per GPU | $32-38/8=$4.00-4.75 per GPU | Widely available |
| GCP | a2-highgpu-8g (8x) | $6-10/8=$0.75-1.25 per GPU | $28-36/8=$3.50-4.50 per GPU | |
| Lambda | gpu_8x_a100 | N/A | $1.10-1.49 per GPU | |
| RunPod | A100 40GB | $0.79 per GPU | $1.39 per GPU | |

## NVIDIA L40S 48GB

Good for training + inference hybrid. More VRAM than A10G, cheaper than A100.

| Cloud | Instance Type | Spot ($/hr) | On-Demand ($/hr) | Notes |
|-------|---------------|-------------|-------------------|-------|
| AWS | g6e.xlarge (1x) | $0.80-1.20 | $2.50-3.50 | Ada Lovelace arch |
| GCP | g2-standard-* | $0.80-1.30 | $2.40-3.80 | |
| Lambda | gpu_1x_l40s | N/A | $0.99 | |
| RunPod | L40S | $0.69 | $1.09 | |

## NVIDIA A10G 24GB

Budget option for fine-tuning small models (up to ~13B with QLoRA) and inference.

| Cloud | Instance Type | Spot ($/hr) | On-Demand ($/hr) | Notes |
|-------|---------------|-------------|-------------------|-------|
| AWS | g5.xlarge (1x) | $0.40-0.60 | $1.50-2.00 | Most available GPU on AWS |
| GCP | N/A | N/A | N/A | Not available on GCP |
| Lambda | gpu_1x_a10 | N/A | $0.75 | |
| RunPod | A10G | $0.29 | $0.49 | |

**Best for:** LoRA/QLoRA fine-tuning, evaluation, small model inference

## NVIDIA T4 16GB

Cheapest option. Good for inference, small experiments, and debugging.

| Cloud | Instance Type | Spot ($/hr) | On-Demand ($/hr) | Notes |
|-------|---------------|-------------|-------------------|-------|
| AWS | g4dn.xlarge (1x) | $0.15-0.25 | $0.53 | Very high spot availability |
| GCP | n1-standard-* + T4 | $0.10-0.20 | $0.35-0.50 | Cheapest T4 option |
| Lambda | N/A | N/A | N/A | |
| RunPod | T4 | $0.10 | $0.20 | |

## NVIDIA H200 141GB

Next-gen after H100. Higher memory bandwidth, more VRAM. Available 2025-2026.

| Cloud | Instance Type | Spot ($/hr) | On-Demand ($/hr) | Notes |
|-------|---------------|-------------|-------------------|-------|
| AWS | p5e (8x) | $22-30/8=$2.75-3.75 per GPU | $90-110/8=$11.25-13.75 per GPU | Limited availability |
| Lambda | gpu_8x_h200 | N/A | $3.49-4.49 per GPU | |
| CoreWeave | H200 | N/A | $3.59 per GPU | |

## Cost Comparison by Workload

### Fine-Tuning 7B Model (QLoRA, ~4 hours)

| GPU | Instance | Spot Cost | On-Demand Cost |
|-----|----------|-----------|----------------|
| A10G (24GB) | AWS g5.xlarge | $1.60-2.40 | $6.00-8.00 |
| A100 40GB | Lambda | N/A | $4.40-5.96 |
| H100 80GB | Lambda | N/A | $9.96-13.16 |

**Winner:** A10G on AWS spot ($1.60-2.40 total)

### Training 13B Model (Full, ~24 hours, 8 GPUs)

| GPU | Instance | Spot Cost | On-Demand Cost |
|-----|----------|-----------|----------------|
| A100 80GB x8 | AWS spot | $192-336 | $960-1,152 |
| A100 80GB x8 | Lambda | N/A | $248-344 |
| H100 80GB x8 | AWS spot | $432-600 | $1,920-2,352 |
| H100 80GB x8 | Lambda | N/A | $479-632 |

**Winner:** Lambda A100 on-demand ($248-344) or AWS A100 spot ($192-336)

### Evaluation Run (~2 hours, single GPU)

| GPU | Instance | Spot Cost | On-Demand Cost |
|-----|----------|-----------|----------------|
| T4 | GCP spot | $0.20-0.40 | $0.70-1.00 |
| A10G | AWS spot | $0.80-1.20 | $3.00-4.00 |
| A100 | Lambda | N/A | $2.58-3.58 |

**Winner:** T4 on GCP spot ($0.20-0.40 total) if model fits in 16GB

## Reserved vs On-Demand vs Spot

| Type | Discount | Commitment | Best For |
|------|----------|------------|----------|
| Spot | 60-75% off | None (can be preempted) | Training with checkpointing |
| On-Demand | Baseline | None | Serving, demos, short jobs |
| 1-Year Reserved | 30-40% off | 1 year prepay | Steady-state training clusters |
| 3-Year Reserved | 50-60% off | 3 year prepay | Long-term research labs |

**Decision rule:**
- Usage < 50% of the month: Spot + On-Demand failover
- Usage 50-80%: 1-Year Reserved
- Usage > 80%: 3-Year Reserved (if confident in GPU choice)

## Spot Preemption Rates by Cloud

Approximate preemption frequency (varies by region and GPU type):

| Cloud | Typical Preemption Interval | Notes |
|-------|----------------------------|-------|
| AWS | 1-6 hours (highly variable) | Use multiple AZs for better odds |
| GCP | 2-24 hours | Preemptible VMs have 24hr max |
| Azure | 1-12 hours | Low-priority VMs |
| Lambda | N/A | No spot offering |

**SkyPilot handles preemption automatically** -- it detects preemption, finds a new spot instance (potentially on a different cloud/region), and resumes from the latest checkpoint.

## Networking Costs (Often Overlooked)

| Item | Cost | Mitigation |
|------|------|------------|
| Data egress (AWS/GCP/Azure) | $0.08-0.12/GB | Keep data and compute in same region |
| Cross-region transfer | $0.01-0.02/GB | Use same region for all nodes |
| S3/GCS storage | $0.023/GB/month | Clean up old checkpoints |
| SkyPilot bucket storage | $0.023/GB/month | Use `sky storage delete` after jobs |

**Rule of thumb:** Networking costs are negligible for training (< 5% of compute cost) but can add up for large dataset transfers. Always colocate data and compute.
