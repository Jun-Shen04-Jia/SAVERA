# SAVERA: Bridging Frozen Vision and Language Models via Sparse Alignment and Visual Evidence Reuse

Official PyTorch training implementation of **SAVERA** (*CVPR 2027 Submission*).

SAVERA connects independently pretrained and frozen vision (`DINOv2-L/14`) and language (`Vicuna-7B-v1.5`) backbones without fine-tuning either foundation model. It consists of three components:
1. **SPARE** (*Shared-space Projection, Activation Reconstruction, and Encoding*): Encodes original visual (`Vision`), language-adapted visual (`Bridge`), and textual (`Text`) features into a shared soft-gated sparse space ($K = 16{,}384$) with branch-specific linear decoders.
2. **BEAM** (*Bridge-Text Embedding and Activation Matching*): Learns a nonlinear `Bridge` mapping and aligns `Bridge` with caption-derived `Text` at both the **sample level** (bidirectional SigLIP matching with a detached FIFO negative queue) and the **neuron level** (symmetric InfoNCE and redundancy-reduction correlation matching over pre-gate activation profiles).
3. **PRIVA** (*Prompt-conditioned Residual Integration of Vision and Adapted Features*): Fuses original `Vision`, adapted `Bridge`, and prompt-derived `Text` via feature-wise sigmoid gating, 2D positional encoding, spatial self-attention, and prompt cross-attention to predict a gated residual correction $\Delta X$ over the frozen `Bridge` visual prefix.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_stage1.sh              # Launch Stage 1 (SPARE-BEAM) training
│   └── run_stage2.sh              # Launch Stage 2 (PRIVA) training
└── savera/
    ├── common/
    │   ├── cuda_utils.py          # Optional CUDA memory/SDPA diagnostics
    │   └── model_loaders.py       # Frozen DINOv2, Vicuna, and Stage-1 checkpoint loaders
    ├── stage1/
    │   └── train.py               # SPARE-BEAM architecture, CC3M loader, and Stage 1 training
    └── stage2/
        ├── data.py                # CC3M prompt/answer collation for Stage 2
        ├── model.py               # PRIVA architecture and auxiliary losses
        ├── runtime.py             # Frozen feature extraction and answer-masked LM loss
        └── train.py               # PRIVA optimization, validation, and checkpointing
```

---

## Installation

Create a Python environment (Python $\ge 3.9$ recommended) with CUDA-enabled PyTorch and install the required packages:

```bash
pip install -r requirements.txt
# Recommended for Vicuna slow tokenizer (use_fast=False)
pip install sentencepiece protobuf
```

---

## Preparation

1. **Pretrained Backbones**: Download local copies of [DINOv2-L/14](https://huggingface.co/facebook/dinov2-large) and [Vicuna-7B-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5). Both models remain completely frozen throughout training.
2. **CC3M Dataset**: Prepare an image directory and an image–caption manifest file. Both stages train exclusively on CC3M image–caption pairs.
   - **Supported manifest formats**: `.tsv`, `.csv`, `.json`, `.jsonl` / `.ndjson`.
   - **Supported columns**: Headered tables with image fields (`image`, `image_path`, `path`, `filename`, etc.) and text fields (`caption`, `text`, `sentence`, `description`, etc.), or headerless 2-column TSV/CSV files. Relative paths, basenames, and extensionless numeric IDs (`.jpg`, `.jpeg`, `.png`, `.webp`) are resolved automatically.

---

## Two-Stage Training

Set the environment variables pointing to your local backbones, CC3M data, and output directories:

```bash
export VISION_MODEL=/path/to/dinov2-large
export LANGUAGE_MODEL=/path/to/vicuna-7b-v1.5
export CC3M_MANIFEST=/path/to/cc3m.tsv
export CC3M_IMAGE_ROOT=/path/to/cc3m_images
```

### Stage 1: Joint SPARE–BEAM Training

Trains the modality-specific stems, shared soft-gated sparse encoder, branch decoders, nonlinear Bridge mapping, and dual-level alignment heads:

```bash
export STAGE1_OUTPUT=./outputs/spare_beam
bash scripts/run_stage1.sh
```

- **Default configuration**: `batch_size=2`, `grad_accum=4`, `lr=2e-4`, `weight_decay=0.01`, cosine schedule with `3%` warmup, `BF16` mixed precision (`DINOv2` evaluated in `FP32`), gate temperature annealing `1.0 -> 0.1` over the first `70%` of steps, target active neurons `64`.
- **Outputs**: Rolling checkpoint `latest.pt` (every `1000` steps) and final checkpoint `final.pt`.

### Stage 2: PRIVA Residual Fusion Training

Freezes the Stage 1 SPARE–BEAM checkpoint and trains PRIVA using prompt-conditioned residual adaptation:

```bash
export STAGE1_CKPT=./outputs/spare_beam/final.pt
export STAGE2_OUTPUT=./outputs/priva
bash scripts/run_stage2.sh
```

- **Default configuration**: `batch_size=2`, `grad_accum=4`, `lr=1e-4`, `weight_decay=0.01`, `max_steps=2500`, `val_fraction=0.03`, fusion width `512` (`8` attention heads), initial residual gate logit $\gamma = -3.0$, initial 2D positional encoding scale $\alpha = 0.1$.
- **Outputs**: Rolling/final checkpoint `latest_priva.pt` and validation metrics `val_history.jsonl`.

### Optional Arguments & Environment Flags

Any extra CLI arguments passed to `run_stage1.sh` or `run_stage2.sh` are forwarded directly to the underlying Python module:

```bash
# Example: Run Stage 2 input-stream ablation (choices: vbt, vb, v_only, b_only)
bash scripts/run_stage2.sh --mode vb

# Example: Resume Stage 2 training from a checkpoint
bash scripts/run_stage2.sh --resume ./outputs/priva/latest_priva.pt
```