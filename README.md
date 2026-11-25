<div align="center">

# SAVERA: Bridging Frozen Vision and Language Models via Sparse Alignment and Visual Evidence Reuse

<p align="center">
  <a>Jiajun Shen</a><sup>1*</sup>,
  <a>Chuancheng Shi</a><sup>2*</sup>,
  <a>Jiayi Lyu</a><sup>1</sup>,
  <a>Yanpeng Sun</a><sup>3</sup>,
  <a>Fei Shen</a><sup>1†</sup>,
  <a>Tat-Seng Chua</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> National University of Singapore, NEXT++ Research Centre,
  <sup>2</sup> University of Sydney,
  <sup>3</sup> Singapore Management University <br>

  <sup>*</sup> Equal contribution &nbsp; | &nbsp; <sup>†</sup> Corresponding author
</p>


[![arXiv](https://img.shields.io/badge/arXiv-2611.32828-b31b1b.svg)](https://arxiv.org/abs/2611.32828)

---
</div>

## Framework Overview

![](figures/pipeline.png)

---

## Installation

```bash
pip install -r requirements.txt
# Recommended for Vicuna slow tokenizer (use_fast=False)
pip install sentencepiece protobuf
```

## Two-Stage Training

Download local copies of [DINOv2-L/14](https://huggingface.co/facebook/dinov2-large), [Vicuna-7B-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5), and the CC3M image–caption dataset, then configure the paths:

```bash
export VISION_MODEL=/path/to/dinov2-large
export LANGUAGE_MODEL=/path/to/vicuna-7b-v1.5
export CC3M_MANIFEST=/path/to/cc3m.tsv
export CC3M_IMAGE_ROOT=/path/to/cc3m_images
```

**Stage 1 (SPARE–BEAM):**

```bash
export STAGE1_OUTPUT=./outputs/spare_beam
bash scripts/run_stage1.sh
```

**Stage 2 (PRIVA):**

```bash
export STAGE1_CKPT=./outputs/spare_beam/final.pt
export STAGE2_OUTPUT=./outputs/priva
bash scripts/run_stage2.sh
```

Any extra CLI arguments passed to `run_stage1.sh` or `run_stage2.sh` are forwarded directly to the underlying Python module:

```bash
# Example: Run Stage 2 input-stream ablation (choices: vbt, vb, v_only, b_only)
bash scripts/run_stage2.sh --mode vb

# Example: Resume Stage 2 training from a checkpoint
bash scripts/run_stage2.sh --resume ./outputs/priva/latest_priva.pt
```

---

> If you have any questions, you can email us at `jiajun.shen@u.nus.edu` or `cshi0459@uni.sydney.edu.au`.
