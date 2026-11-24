#!/usr/bin/env python3
"""Stage 1: SPARE shared sparse encoding and BEAM cross-modal alignment on CC3M."""

import argparse
import csv
import json
import math
import os
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ----------------------------- data -----------------------------------------

IMAGE_KEYS = (
    "image", "image_path", "path", "filepath", "file", "filename",
    "local_path", "jpg_path", "image_file", "file_name", "img", "jpg",
)
TEXT_KEYS = ("caption", "text", "sentence", "description", "title", "txt", "prompt")


def _first_present(d: Dict, keys: Sequence[str]):
    # Accept common case/capitalization differences in local CC3M manifests.
    lower = {str(k).strip().lower(): v for k, v in d.items()}
    for k in keys:
        v = lower.get(str(k).lower(), None)
        if v not in (None, ""):
            return v
    return None


def read_manifest(path: str) -> List[Tuple[str, str]]:
    """Read JSONL / JSON / TSV / CSV with flexible image/caption column names."""
    p = Path(path)
    suf = p.suffix.lower()
    rows: List[Tuple[str, str]] = []

    if suf in (".jsonl", ".ndjson"):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                im = _first_present(obj, IMAGE_KEYS)
                tx = _first_present(obj, TEXT_KEYS)
                if im is not None and tx is not None:
                    rows.append((str(im), str(tx)))
        return rows

    if suf == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            obj = obj.get("data", obj.get("annotations", obj.get("items", [])))
        for item in obj:
            im = _first_present(item, IMAGE_KEYS)
            tx = _first_present(item, TEXT_KEYS)
            if im is not None and tx is not None:
                rows.append((str(im), str(tx)))
        return rows

    delim = "\t" if suf in (".tsv", ".txt") else ","
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        fields = [str(x).strip().lower() for x in (reader.fieldnames or [])]
        has_image_header = any(k.lower() in fields for k in IMAGE_KEYS)
        has_text_header = any(k.lower() in fields for k in TEXT_KEYS)

        if has_image_header and has_text_header:
            for item in reader:
                im = _first_present(item, IMAGE_KEYS)
                tx = _first_present(item, TEXT_KEYS)
                if im is not None and tx is not None:
                    rows.append((str(im), str(tx)))
        else:
            # Headerless two-column CC3M pair files are common.  DictReader would
            # otherwise silently consume the first data row as a header.
            f.seek(0)
            raw = csv.reader(f, delimiter=delim)
            for item in raw:
                if len(item) < 2:
                    continue
                a, b = str(item[0]).strip(), str(item[1]).strip()
                if not a or not b:
                    continue
                # Skip an unrecognised textual header if present.
                if a.lower() in IMAGE_KEYS or b.lower() in TEXT_KEYS:
                    continue
                rows.append((a, b))
    return rows


COMMON_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP")


def _candidate_image_paths(rel: str, image_root: Optional[Path]):
    """Generate plausible local paths for a CC3M manifest image field.

    Local CC3M pair files often store a bare numeric id such as ``000000123``
    while the extracted image is ``cc3m_jpg/000000123.jpg``.  Other exports
    store a filename, relative path, or absolute path.  Keep the common cases
    cheap and deterministic; do not recursively glob millions of files.
    """
    raw = str(rel).strip()
    q = Path(raw)

    bases = []
    if q.is_absolute():
        bases.append(q)
    else:
        if image_root is not None:
            bases.append(image_root / q)
            # Handle manifests containing e.g. ``cc3m_jpg/000000123`` while
            # image_root already points at ``.../cc3m_jpg``.
            if q.name != raw:
                bases.append(image_root / q.name)
        else:
            bases.append(q)

    # Preserve order while deduplicating.
    uniq = []
    seen = set()
    for b in bases:
        key = str(b)
        if key not in seen:
            uniq.append(b)
            seen.add(key)

    for b in uniq:
        yield b
        # Bare ids / extensionless local paths are by far the most common case.
        if b.suffix == "":
            for ext in COMMON_IMAGE_EXTS:
                yield Path(str(b) + ext)

        # A few CC3M exports convert numeric ids to a different zero-padding.
        stem = b.name
        if b.suffix:
            stem = b.stem
        if stem.isdigit():
            n = int(stem)
            for width in (8, 9, 10, 12):
                padded = f"{n:0{width}d}"
                pb = b.with_name(padded)
                yield pb
                for ext in COMMON_IMAGE_EXTS:
                    yield Path(str(pb) + ext)


def resolve_image_path(rel: str, image_root: Optional[Path]) -> Optional[Path]:
    for p in _candidate_image_paths(rel, image_root):
        if p.is_file():
            return p
    return None


class CC3MPairedDataset(Dataset):
    def __init__(self, manifest: str, image_root: str = "", max_samples: int = 0):
        self.image_root = Path(image_root) if image_root else None
        self.rows = read_manifest(manifest)
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise RuntimeError(
                f"No usable image-caption pairs found in {manifest}. "
                "Expected a header containing an image/path field and a caption/text field, "
                "or a headerless 2-column TSV/CSV."
            )

        # Detect the local filename convention once.  For a bare-id manifest,
        # this normally discovers `.jpg` immediately and then __getitem__ avoids
        # repeated filesystem probing for millions of samples.
        probe_n = min(64, len(self.rows))
        resolved_probe = []
        for rel, _ in self.rows[:probe_n]:
            resolved_probe.append(resolve_image_path(rel, self.image_root))

        found = sum(p is not None for p in resolved_probe)
        if found == 0:
            rel_examples = [str(x[0]) for x in self.rows[:3]]
            tried = []
            for rel in rel_examples:
                tried.append([str(p) for _, p in zip(range(8), _candidate_image_paths(rel, self.image_root))])
            raise RuntimeError(
                f"Manifest parsed {len(self.rows)} pairs, but none of the first {probe_n} image ids could be resolved. "
                f"image_root={self.image_root}; manifest_examples={rel_examples}; tried={tried}"
            )

        # Infer a fast transformation from manifest field -> local path.
        # We support the common root/relative + suffix and root/basename + suffix forms.
        mode_counts = {}
        for (rel, _), rp in zip(self.rows[:probe_n], resolved_probe):
            if rp is None:
                continue
            q = Path(str(rel).strip())
            suffix = rp.suffix
            if q.is_absolute():
                mode = ("absolute", suffix if q.suffix == "" else "")
            elif self.image_root is not None:
                direct = self.image_root / q
                basename = self.image_root / q.name
                if rp == direct:
                    mode = ("root_rel", "")
                elif q.suffix == "" and rp == Path(str(direct) + suffix):
                    mode = ("root_rel", suffix)
                elif rp == basename:
                    mode = ("root_base", "")
                elif q.suffix == "" and rp == Path(str(basename) + suffix):
                    mode = ("root_base", suffix)
                else:
                    mode = ("fallback", "")
            else:
                mode = ("relative", suffix if q.suffix == "" else "")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        self.path_mode = max(mode_counts, key=mode_counts.get)
        self.path_mode_hits = int(mode_counts[self.path_mode])
        self.path_probe_found = int(found)

        # Small cache for fallback / repeatedly sampled records.  Do not resolve
        # all ~3M entries eagerly; that would perform millions of stat calls.
        self._resolved_cache = {}
        for i, rp in enumerate(resolved_probe):
            if rp is not None:
                self._resolved_cache[i] = str(rp)

        print(json.dumps({
            "cc3m_manifest": str(manifest),
            "cc3m_pairs": len(self.rows),
            "cc3m_image_root": str(self.image_root),
            "cc3m_path_mode": list(self.path_mode),
            "probe_resolved": f"{found}/{probe_n}",
            "path_mode_hits": self.path_mode_hits,
            "first_manifest_image": str(self.rows[0][0]),
            "first_resolved_image": str(resolved_probe[0]) if resolved_probe[0] is not None else None,
        }, indent=2), flush=True)

    def __len__(self):
        return len(self.rows)

    def _fast_path(self, rel: str) -> Path:
        kind, add_suffix = self.path_mode
        q = Path(str(rel).strip())
        if kind == "absolute":
            p = q
        elif kind == "root_rel":
            p = self.image_root / q
        elif kind == "root_base":
            p = self.image_root / q.name
        elif kind == "relative":
            p = q
        else:
            p = self.image_root / q if self.image_root is not None and not q.is_absolute() else q
        if add_suffix and p.suffix == "":
            p = Path(str(p) + add_suffix)
        return p

    def __getitem__(self, idx):
        rel, caption = self.rows[idx]
        if idx in self._resolved_cache:
            return self._resolved_cache[idx], caption

        p = self._fast_path(rel)
        if not p.is_file():
            rp = resolve_image_path(rel, self.image_root)
            if rp is None:
                # Let the error identify the exact bad row instead of PIL emitting
                # a less informative FileNotFoundError from a worker.
                candidates = [str(x) for _, x in zip(range(10), _candidate_image_paths(rel, self.image_root))]
                raise RuntimeError(
                    f"Could not resolve CC3M image at dataset index={idx}, manifest_value={rel!r}; "
                    f"tried={candidates}"
                )
            p = rp
        return str(p), caption


def expand_to_square(img: Image.Image, fill_rgb: Tuple[int, int, int]) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    out = Image.new("RGB", (side, side), fill_rgb)
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out


class CC3MCollator:
    def __init__(self, tokenizer, image_mean, image_std, image_size=336, max_text_len=64, pad_square=True):
        self.tokenizer = tokenizer
        self.mean = tuple(float(x) for x in image_mean)
        self.std = tuple(float(x) for x in image_std)
        self.image_size = int(image_size)
        self.max_text_len = int(max_text_len)
        self.pad_square = bool(pad_square)
        self.fill_rgb = tuple(int(round(255 * x)) for x in self.mean)

    def _image_tensor(self, path: str):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {path}: {e}")
        if self.pad_square:
            img = expand_to_square(img, self.fill_rgb)
        # Force EXACT spatial size.  This is important for DINO: 336/14=24 => 576 patches.
        try:
            img = TF.resize(
                img, [self.image_size, self.image_size],
                interpolation=InterpolationMode.BICUBIC, antialias=True
            )
        except TypeError:
            # torchvision versions paired with some Python-3.8 environments do
            # not expose the antialias keyword yet.
            img = TF.resize(
                img, [self.image_size, self.image_size],
                interpolation=InterpolationMode.BICUBIC
            )
        x = TF.to_tensor(img)
        x = TF.normalize(x, self.mean, self.std)
        return x

    def __call__(self, batch):
        paths, captions = zip(*batch)
        pixels = torch.stack([self._image_tensor(p) for p in paths], dim=0)
        toks = self.tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixels,
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "captions": list(captions),
            "paths": list(paths),
        }


# ----------------------------- sparse modules -------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class SoftGatedSharedEncoder(nn.Module):
    """One shared SAE encoder with differentiable soft gating.

    a = W_E RMSNorm(x) + b
    gate = sigmoid((a - theta) / temperature)
    magnitude = softplus(a)
    z = gate * magnitude

    No sample-wise TopK is used.  Sparsity is controlled with expected-active
    budget + gate-binarization losses and temperature annealing.
    """

    def __init__(self, in_dim: int, latent_dim: int, init_threshold: float = 2.0):
        super().__init__()
        self.norm = RMSNorm(in_dim)
        self.encoder = nn.Linear(in_dim, latent_dim)
        self.threshold = nn.Parameter(torch.full((latent_dim,), float(init_threshold)))
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))

    def forward(self, x, temperature: float):
        a = self.encoder(self.norm(x))
        gate = torch.sigmoid((a - self.threshold) / max(float(temperature), 1e-4))
        magnitude = F.softplus(a)
        z = gate * magnitude
        return z, a, gate


class VisionStem(nn.Module):
    def __init__(self, vision_dim=1024, canonical_dim=4096):
        super().__init__()
        self.norm = RMSNorm(vision_dim)
        self.proj = nn.Linear(vision_dim, canonical_dim)

    def forward(self, x):
        return self.proj(self.norm(x))


class TextStem(nn.Module):
    def __init__(self, dim=4096):
        super().__init__()
        self.norm = RMSNorm(dim)
        # Small residual adapter instead of a second full 4096x4096 projection.
        self.scale = nn.Parameter(torch.zeros(()))
        self.adapter = nn.Sequential(
            nn.Linear(dim, 512, bias=False),
            nn.SiLU(),
            nn.Linear(512, dim, bias=False),
        )

    def forward(self, x):
        y = self.norm(x)
        return y + self.scale * self.adapter(y)


class BridgeInputMLP(nn.Module):
    """Translate vision sparse coordinates into canonical 4096-D bridge input."""

    def __init__(self, latent_dim=16384, canonical_dim=4096, bottleneck=2048):
        super().__init__()
        self.norm = RMSNorm(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, canonical_dim, bias=True),
        )

    def forward(self, z):
        return self.net(self.norm(z))


class SPAREBEAM(nn.Module):
    def __init__(
        self,
        vision_dim=1024,
        canonical_dim=4096,
        latent_dim=16384,
        bridge_bottleneck=2048,
        align_dim=1024,
        init_threshold=2.0,
    ):
        super().__init__()
        self.vision_stem = VisionStem(vision_dim, canonical_dim)
        self.text_stem = TextStem(canonical_dim)
        self.shared_encoder = SoftGatedSharedEncoder(canonical_dim, latent_dim, init_threshold)

        self.vision_decoder = nn.Linear(latent_dim, vision_dim)
        self.bridge_in = BridgeInputMLP(latent_dim, canonical_dim, bridge_bottleneck)
        self.bridge_decoder = nn.Linear(latent_dim, canonical_dim)
        self.text_decoder = nn.Linear(latent_dim, canonical_dim)

        # Shared semantic head: same latent coordinates, same head.
        self.align_head = nn.Linear(latent_dim, align_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))

    def encode_vision(self, h_v, temperature):
        u_v = self.vision_stem(h_v)
        z_v, a_v, g_v = self.shared_encoder(u_v, temperature)
        h_v_hat = self.vision_decoder(z_v)
        return {"u": u_v, "z": z_v, "a": a_v, "gate": g_v, "recon": h_v_hat}

    def encode_bridge(self, z_v, temperature):
        u_b = self.bridge_in(z_v)
        z_b, a_b, g_b = self.shared_encoder(u_b, temperature)
        x_prime = self.bridge_decoder(z_b)
        return {"u": u_b, "z": z_b, "a": a_b, "gate": g_b, "recon": x_prime, "x_prime": x_prime}

    def encode_text(self, h_t, temperature):
        u_t = self.text_stem(h_t)
        z_t, a_t, g_t = self.shared_encoder(u_t, temperature)
        h_t_hat = self.text_decoder(z_t)
        return {"u": u_t, "z": z_t, "a": a_t, "gate": g_t, "recon": h_t_hat}

    def forward(self, h_v, h_t, text_mask, temperature):
        # Keep all trainable parameter usage inside DDP/Accelerate forward.
        V = self.encode_vision(h_v, temperature)
        B = self.encode_bridge(V["z"], temperature)
        T = self.encode_text(h_t, temperature)

        zb_pool = B["z"].mean(dim=1)
        w = text_mask.to(T["z"].dtype).unsqueeze(-1)
        zt_pool = (T["z"] * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        sem_b = F.normalize(self.align_head(zb_pool.float()), dim=-1)
        sem_t = F.normalize(self.align_head(zt_pool.float()), dim=-1)
        return V, B, T, sem_b, sem_t, self.logit_scale, self.logit_bias


# ----------------------------- losses ---------------------------------------


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None):
    if mask is None:
        return x.mean(dim=1)
    w = mask.to(x.dtype).unsqueeze(-1)
    return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


def normalized_reconstruction_loss(pred, target, mask=None, cosine_weight=0.2):
    if mask is None:
        p = pred.reshape(-1, pred.shape[-1]).float()
        t = target.reshape(-1, target.shape[-1]).float()
    else:
        m = mask.bool().reshape(-1)
        p = pred.reshape(-1, pred.shape[-1])[m].float()
        t = target.reshape(-1, target.shape[-1])[m].float()
    if p.numel() == 0:
        return pred.sum() * 0.0
    nmse = (p - t).pow(2).sum(dim=-1) / t.pow(2).sum(dim=-1).clamp_min(1e-6)
    cos = 1.0 - F.cosine_similarity(p, t, dim=-1)
    return nmse.mean() + float(cosine_weight) * cos.mean()


def soft_sparsity_loss(gates: List[torch.Tensor], target_active: float):
    losses = []
    stats = []
    for g in gates:
        # g shape [B, tokens, K]
        expected_active = g.float().sum(dim=-1)
        budget = ((expected_active.mean() - target_active) / max(target_active, 1.0)).pow(2)
        # Encourage gates toward 0/1 while keeping fully differentiable forward.
        binary = (g.float() * (1.0 - g.float())).mean()
        l0_proxy = g.float().mean()
        hard_active = (g.detach() > 0.5).float().sum(dim=-1).mean()
        losses.append((budget, binary, l0_proxy))
        stats.append((expected_active.mean().detach(), hard_active.detach()))
    budget = torch.stack([x[0] for x in losses]).mean()
    binary = torch.stack([x[1] for x in losses]).mean()
    l0_proxy = torch.stack([x[2] for x in losses]).mean()
    exp_active = torch.stack([x[0] for x in stats]).mean()
    hard_active = torch.stack([x[1] for x in stats]).mean()
    return budget, binary, l0_proxy, exp_active, hard_active


class SemanticBank:
    """Detached FIFO negatives for SigLIP-style alignment (works with batch=1)."""

    def __init__(self, max_samples=512):
        self.max_samples = int(max_samples)
        self.b = deque()
        self.t = deque()

    def add(self, vb: torch.Tensor, vt: torch.Tensor):
        for bi, ti in zip(vb.detach().cpu().float(), vt.detach().cpu().float()):
            self.b.append(bi)
            self.t.append(ti)
            while len(self.b) > self.max_samples:
                self.b.popleft(); self.t.popleft()

    def tensors(self, device):
        if not self.b:
            return None, None
        b = torch.stack(list(self.b), dim=0).to(device=device)
        t = torch.stack(list(self.t), dim=0).to(device=device)
        return b, t


def siglip_pair_loss(vb, vt, logit_scale, logit_bias, bank: SemanticBank):
    scale = logit_scale.exp().clamp(max=100.0)
    old_b, old_t = bank.tensors(vb.device)

    # current visual -> current positive text + current/queue negatives
    all_t = vt if old_t is None else torch.cat([vt, old_t], dim=0)
    logits_bt = scale * (vb @ all_t.t()) + logit_bias
    labels_bt = -torch.ones_like(logits_bt)
    diag = torch.arange(vb.shape[0], device=vb.device)
    labels_bt[diag, diag] = 1.0
    loss_bt = F.softplus(-labels_bt * logits_bt).mean()

    # symmetric current text -> current positive visual + current/queue negatives
    all_b = vb if old_b is None else torch.cat([vb, old_b], dim=0)
    logits_tb = scale * (vt @ all_b.t()) + logit_bias
    labels_tb = -torch.ones_like(logits_tb)
    labels_tb[diag, diag] = 1.0
    loss_tb = F.softplus(-labels_tb * logits_tb).mean()
    return 0.5 * (loss_bt + loss_tb)


class ProfileBank:
    """CPU bank of pooled bridge/text pre-sparsity activations for neuron profiles."""

    def __init__(self, max_samples=256):
        self.max_samples = int(max_samples)
        self.b = deque()
        self.t = deque()
        self.n = 0

    def add(self, b: torch.Tensor, t: torch.Tensor):
        b = b.detach().to("cpu", dtype=torch.float16)
        t = t.detach().to("cpu", dtype=torch.float16)
        for bi, ti in zip(b, t):
            self.b.append(bi)
            self.t.append(ti)
            self.n += 1
            while self.n > self.max_samples:
                self.b.popleft(); self.t.popleft(); self.n -= 1

    def get_columns(self, idx: torch.Tensor, device):
        if self.n == 0:
            return None, None
        ids = idx.detach().cpu()
        b = torch.stack([x.index_select(0, ids) for x in self.b], dim=0).to(device=device, dtype=torch.float32)
        t = torch.stack([x.index_select(0, ids) for x in self.t], dim=0).to(device=device, dtype=torch.float32)
        return b, t


def neuron_profile_losses(
    a_b_pool: torch.Tensor,
    a_t_pool: torch.Tensor,
    bank: ProfileBank,
    sample_neurons: int,
    temperature: float,
):
    """Same-index neuron InfoNCE + Barlow-style correlation identity.

    Profiles are activation trajectories across paired CC3M samples.  Historical
    samples are detached context; current batch remains differentiable.
    """
    B, K = a_b_pool.shape
    q = min(int(sample_neurons), K)
    # Prefer high-variance current neurons, fill remainder randomly.
    with torch.no_grad():
        variance = 0.5 * (a_b_pool.float().var(dim=0, unbiased=False) + a_t_pool.float().var(dim=0, unbiased=False))
        q_hi = min(q // 2, K)
        hi = torch.topk(variance, k=q_hi, largest=True).indices if q_hi > 0 else torch.empty(0, dtype=torch.long, device=a_b_pool.device)
        if q_hi < q:
            rand = torch.randperm(K, device=a_b_pool.device)
            mask = torch.ones(K, dtype=torch.bool, device=a_b_pool.device)
            mask[hi] = False
            rand = rand[mask[rand]][: q - q_hi]
            idx = torch.cat([hi, rand], dim=0)
        else:
            idx = hi

    cur_b = a_b_pool.float().index_select(1, idx)
    cur_t = a_t_pool.float().index_select(1, idx)
    old_b, old_t = bank.get_columns(idx, a_b_pool.device)
    prof_b = cur_b if old_b is None else torch.cat([old_b, cur_b], dim=0)
    prof_t = cur_t if old_t is None else torch.cat([old_t, cur_t], dim=0)

    if prof_b.shape[0] < 4:
        zero = a_b_pool.sum() * 0.0
        return zero, zero, zero

    # neuron-profile InfoNCE: q neuron profiles x q neuron profiles.
    pb = F.normalize(prof_b.t(), dim=-1)
    pt = F.normalize(prof_t.t(), dim=-1)
    sim = (pb @ pt.t()) / max(float(temperature), 1e-4)
    target = torch.arange(q, device=sim.device)
    nce = 0.5 * (F.cross_entropy(sim, target) + F.cross_entropy(sim.t(), target))

    # Correlation identity C ~= I.
    xb = prof_b - prof_b.mean(dim=0, keepdim=True)
    xt = prof_t - prof_t.mean(dim=0, keepdim=True)
    xb = xb / xb.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    xt = xt / xt.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    C = (xb.t() @ xt) / float(prof_b.shape[0])
    diag = torch.diagonal(C)
    diag_loss = (1.0 - diag).pow(2).mean()
    off = C - torch.diag_embed(diag)
    off_loss = off.pow(2).mean()
    return nce, diag_loss, off_loss


def bridge_lm_loss(llm, x_prime, input_ids, attention_mask):
    """Frozen Vicuna causal loss with x' as 4096-D visual-prefix tokens."""
    B, Nv, D = x_prime.shape
    emb_layer = llm.get_input_embeddings()
    text_emb = emb_layer(input_ids)
    # Ensure dtype matches the frozen LLM while preserving grad from x_prime.
    xp = x_prime.to(dtype=text_emb.dtype)
    inputs_embeds = torch.cat([xp, text_emb], dim=1)
    visual_mask = torch.ones(B, Nv, device=attention_mask.device, dtype=attention_mask.dtype)
    full_mask = torch.cat([visual_mask, attention_mask], dim=1)
    ignore = torch.full((B, Nv), -100, device=input_ids.device, dtype=input_ids.dtype)
    text_labels = input_ids.masked_fill(attention_mask == 0, -100)
    labels = torch.cat([ignore, text_labels], dim=1)
    out = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        labels=labels,
        use_cache=False,
        return_dict=True,
    )
    return out.loss


# ----------------------------- frozen towers --------------------------------


def freeze_module(m: nn.Module):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)


@torch.no_grad()
def extract_vision_tokens(vision_model, backend: str, pixel_values, layer_index: int):
    out = vision_model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[layer_index]
    # Exclude the DINOv2 class token and retain the 576 image patches.
    h = h[:, 1:, :]
    return h


@torch.no_grad()
def extract_text_hidden(llm, input_ids, attention_mask, text_layer: int):
    out = llm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return out.hidden_states[text_layer]


def temperature_schedule(step: int, max_steps: int, start: float, end: float, anneal_frac: float):
    if max_steps <= 0:
        return end
    T = max(1, int(max_steps * anneal_frac))
    f = min(max(step / T, 0.0), 1.0)
    # Log-space annealing is smoother for sigmoid temperatures.
    return float(math.exp(math.log(start) * (1 - f) + math.log(end) * f))


# ----------------------------- train ----------------------------------------


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--vision_backend", choices=["dino"], default="dino")
    p.add_argument("--vision_model", required=True)
    p.add_argument("--text_model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--image_root", default="")
    p.add_argument("--output_dir", required=True)

    p.add_argument("--image_size", type=int, default=336)
    p.add_argument("--vision_layer", type=int, default=-2)
    p.add_argument("--text_layer", type=int, default=16)
    p.add_argument("--max_text_len", type=int, default=64)
    # Python 3.8 compatible replacement for argparse.BooleanOptionalAction (3.9+).
    p.add_argument("--pad_square", dest="pad_square", action="store_true")
    p.add_argument("--no_pad_square", dest="pad_square", action="store_false")
    p.set_defaults(pad_square=True)

    p.add_argument("--vision_dim", type=int, default=1024)
    p.add_argument("--canonical_dim", type=int, default=4096)
    p.add_argument("--latent_dim", type=int, default=16384)
    p.add_argument("--bridge_bottleneck", type=int, default=2048)
    p.add_argument("--align_dim", type=int, default=1024)

    p.add_argument("--target_active", type=float, default=64.0)
    p.add_argument("--gate_temp_start", type=float, default=1.0)
    p.add_argument("--gate_temp_end", type=float, default=0.10)
    p.add_argument("--gate_anneal_frac", type=float, default=0.70)
    p.add_argument("--init_threshold", type=float, default=2.0)

    p.add_argument("--lambda_vrec", type=float, default=1.0)
    p.add_argument("--lambda_brec", type=float, default=0.25)
    p.add_argument("--lambda_trec", type=float, default=1.0)
    p.add_argument("--lambda_budget", type=float, default=1.0)
    p.add_argument("--lambda_binary", type=float, default=0.05)
    p.add_argument("--lambda_l0", type=float, default=0.01)
    p.add_argument("--lambda_sem", type=float, default=0.20)
    p.add_argument("--lambda_neuron_nce", type=float, default=0.10)
    p.add_argument("--lambda_corr_diag", type=float, default=0.05)
    p.add_argument("--lambda_corr_off", type=float, default=0.01)
    p.add_argument("--lambda_lm", type=float, default=1.0)
    p.add_argument("--recon_cosine_weight", type=float, default=0.20)
    p.add_argument("--neuron_profile_temp", type=float, default=0.10)
    p.add_argument("--profile_neurons", type=int, default=512)
    p.add_argument("--profile_bank_samples", type=int, default=256)
    p.add_argument("--semantic_bank_samples", type=int, default=512)
    p.add_argument("--lm_every", type=int, default=1, help="Compute expensive frozen-Vicuna LM loss every N microsteps")

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    p.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true")
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    p.set_defaults(gradient_checkpointing=True)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p


def public_args(args):
    """Exclude local file paths from Stage-1 checkpoint metadata."""
    hidden={'vision_model','text_model','manifest','image_root','output_dir'}
    return {k: ('<local-path>' if k in hidden and v else v) for k,v in vars(args).items()}


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device
    outdir = Path(args.output_dir)
    if accelerator.is_main_process:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "args.json").write_text(json.dumps(public_args(args), indent=2), encoding="utf-8")
    accelerator.wait_for_everyone()

    # ---- fail-fast CC3M manifest/path validation BEFORE loading 7B models ----
    ds = CC3MPairedDataset(args.manifest, args.image_root, args.max_samples)
    if accelerator.is_main_process:
        rel0, cap0 = ds.rows[0]
        p0 = Path(rel0)
        if not p0.is_absolute() and ds.image_root is not None:
            p0 = ds.image_root / p0
        print(json.dumps({
            "dataset_pairs": len(ds),
            "manifest": args.manifest,
            "image_root": args.image_root,
            "first_image": str(p0),
            "first_image_exists": p0.is_file(),
            "first_caption": cap0[:120],
            "python": __import__("sys").version.split()[0],
        }, ensure_ascii=False), flush=True)

    # ---- tokenizer / frozen Vicuna ----
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    frozen_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (torch.float16 if args.mixed_precision == "fp16" else torch.float32)
    llm = AutoModelForCausalLM.from_pretrained(
        args.text_model,
        torch_dtype=frozen_dtype,
        low_cpu_mem_usage=True,
    )
    llm.config.use_cache = False
    freeze_module(llm)
    if args.gradient_checkpointing:
        try:
            llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            llm.gradient_checkpointing_enable()
    llm.to(device)

    # ---- frozen vision tower + normalization ----
    processor = AutoImageProcessor.from_pretrained(args.vision_model)
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])

    # DINOv2 interpolation at 336x336 is performed in FP32.
    vision_dtype = torch.float32
    vision = AutoModel.from_pretrained(args.vision_model, torch_dtype=vision_dtype)
    freeze_module(vision)
    vision.to(device)

    if accelerator.is_main_process:
        print(json.dumps({
            "vision_backend": args.vision_backend,
            "vision_dtype": str(vision_dtype),
            "llm_dtype": str(frozen_dtype),
            "image_size": args.image_size,
        }), flush=True)

    # ---- data ----
    collate = CC3MCollator(
        tokenizer=tokenizer,
        image_mean=mean,
        image_std=std,
        image_size=args.image_size,
        max_text_len=args.max_text_len,
        pad_square=args.pad_square,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )

    # ---- trainable model ----
    model = SPAREBEAM(
        vision_dim=args.vision_dim,
        canonical_dim=args.canonical_dim,
        latent_dim=args.latent_dim,
        bridge_bottleneck=args.bridge_bottleneck,
        align_dim=args.align_dim,
        init_threshold=args.init_threshold,
    )

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or "threshold" in n or "bias" in n or "norm" in n) else decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    steps_per_epoch = math.ceil(len(dl) / max(args.grad_accum, 1))
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model, optimizer, dl, scheduler = accelerator.prepare(model, optimizer, dl, scheduler)
    profile_bank = ProfileBank(args.profile_bank_samples)
    semantic_bank = SemanticBank(args.semantic_bank_samples)

    global_step = 0
    microstep = 0
    running = {}

    for epoch in range(args.epochs):
        model.train()
        for batch in dl:
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
            microstep += 1
            # DINOv2@336 requires FP32 specifically for positional-embedding
            # bicubic interpolation on the current PyTorch stack.
            pixels = batch["pixel_values"].to(
                device, non_blocking=True, dtype=vision_dtype
            )
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)

            # Frozen vision features are taken from the selected DINOv2 hidden layer.
            h_v = extract_vision_tokens(vision, args.vision_backend, pixels, args.vision_layer)
            if h_v.shape[-1] != args.vision_dim:
                raise RuntimeError(f"vision dim mismatch: got {h_v.shape[-1]}, expected {args.vision_dim}")
            if args.image_size == 336 and h_v.shape[1] != 576:
                raise RuntimeError(
                    f"Expected 576 patch tokens at 336px/patch14, got {h_v.shape[1]}. "
                    "Check vision model patch size / preprocessing."
                )
            h_t = extract_text_hidden(llm, input_ids, attn, args.text_layer)
            if h_t.shape[-1] != args.canonical_dim:
                raise RuntimeError(f"text dim mismatch: got {h_t.shape[-1]}, expected {args.canonical_dim}")

            temp = temperature_schedule(global_step, total_steps, args.gate_temp_start, args.gate_temp_end, args.gate_anneal_frac)

            with accelerator.accumulate(model):
                # Three sparse modules sharing exactly the same encoder weights.
                V, B, T, sem_b, sem_t, sem_scale, sem_bias = model(h_v, h_t, attn, temp)

                loss_vrec = normalized_reconstruction_loss(V["recon"], h_v, cosine_weight=args.recon_cosine_weight)
                # Bridge SAE reconstructs its learned canonical bridge input u_B.
                # Detach bridge input target so the learned bridge input MLP cannot collapse
                # its own target merely to reduce reconstruction loss.
                loss_brec = normalized_reconstruction_loss(B["recon"], B["u"].detach(), cosine_weight=args.recon_cosine_weight)
                loss_trec = normalized_reconstruction_loss(T["recon"], h_t, mask=attn, cosine_weight=args.recon_cosine_weight)

                budget, binary, l0_proxy, exp_active, hard_active = soft_sparsity_loss(
                    [V["gate"], B["gate"], T["gate"]], args.target_active
                )

                # Sample-level multimodal alignment uses pooled sparse latents.
                loss_sem = siglip_pair_loss(sem_b, sem_t, sem_scale, sem_bias, semantic_bank)

                # Neuron identity uses FULL PRE-SPARSITY activation profiles, not sample-wise hard supports.
                ab_pool = masked_mean(B["a"])
                at_pool = masked_mean(T["a"], attn)
                loss_nce, loss_diag, loss_off = neuron_profile_losses(
                    ab_pool,
                    at_pool,
                    profile_bank,
                    args.profile_neurons,
                    args.neuron_profile_temp,
                )

                # Functional objective.  Bridge representations are learned through the training objectives.
                do_lm = args.lambda_lm > 0 and (microstep % max(args.lm_every, 1) == 0)
                if do_lm:
                    loss_lm = bridge_lm_loss(llm, B["x_prime"], input_ids, attn)
                    # If LM is computed less frequently, rescale to preserve expected contribution.
                    loss_lm_scaled = loss_lm * float(max(args.lm_every, 1))
                else:
                    loss_lm = B["x_prime"].sum() * 0.0
                    loss_lm_scaled = loss_lm

                loss = (
                    args.lambda_vrec * loss_vrec
                    + args.lambda_brec * loss_brec
                    + args.lambda_trec * loss_trec
                    + args.lambda_budget * budget
                    + args.lambda_binary * binary
                    + args.lambda_l0 * l0_proxy
                    + args.lambda_sem * loss_sem
                    + args.lambda_neuron_nce * loss_nce
                    + args.lambda_corr_diag * loss_diag
                    + args.lambda_corr_off * loss_off
                    + args.lambda_lm * loss_lm_scaled
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Add profiles AFTER using current bank; bank entries are detached CPU fp16.
            profile_bank.add(ab_pool, at_pool)
            semantic_bank.add(sem_b, sem_t)

            vals = {
                "loss": loss.detach(),
                "vrec": loss_vrec.detach(),
                "brec": loss_brec.detach(),
                "trec": loss_trec.detach(),
                "budget": budget.detach(),
                "binary": binary.detach(),
                "sem": loss_sem.detach(),
                "nce": loss_nce.detach(),
                "diag": loss_diag.detach(),
                "off": loss_off.detach(),
                "lm": loss_lm.detach(),
                "exp_active": exp_active.detach(),
                "hard_active": hard_active.detach(),
            }
            for k, v in vals.items():
                running[k] = running.get(k, 0.0) + float(accelerator.gather(v.float().reshape(1)).mean().item())

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % args.log_every == 0:
                    denom = float(args.log_every * args.grad_accum)
                    msg = {k: running[k] / denom for k in running}
                    msg.update({"step": global_step, "temp": temp, "lr": scheduler.get_last_lr()[0]})
                    print(json.dumps(msg, sort_keys=True), flush=True)
                    running = {}

                if global_step % args.save_every == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        raw = accelerator.unwrap_model(model)
                        ck = {
                            "model": raw.state_dict(),
                            "args": public_args(args),
                            "step": global_step,
                            "epoch": epoch,
                        }
                        # Keep only ONE rolling checkpoint to avoid filling disk.
                        # Write atomically: if the job dies during torch.save, the previous
                        # latest.pt remains intact.
                        tmp_ckpt = outdir / "latest.tmp.pt"
                        latest_ckpt = outdir / "latest.pt"
                        torch.save(ck, tmp_ckpt)
                        os.replace(tmp_ckpt, latest_ckpt)
                        print(f"saved latest checkpoint: {latest_ckpt} (step={global_step})", flush=True)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        raw = accelerator.unwrap_model(model)
        torch.save(
            {"model": raw.state_dict(), "args": public_args(args), "step": global_step, "epoch": args.epochs},
            outdir / "final.pt",
        )
        print(f"saved: {outdir / 'final.pt'}")


if __name__ == "__main__":
    main()
