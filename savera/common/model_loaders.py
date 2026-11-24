#!/usr/bin/env python3
"""Load frozen SPARE--BEAM and pretrained backbone models for PRIVA."""
from pathlib import Path

import torch


def torch_load(p):
    try:
        return torch.load(str(p), map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(str(p), map_location='cpu')


def file_fingerprint(path):
    p=Path(path).resolve();st=p.stat()
    return {'size':st.st_size,'mtime_ns':st.st_mtime_ns}


def confirm_sae_fingerprint(reference, current):
    if reference is None:return
    if reference.get('size')!=current['size'] or reference.get('mtime_ns')!=current['mtime_ns']:
        raise RuntimeError('The Stage-1 SPARE--BEAM checkpoint has changed since PRIVA training.'
                           'Use the Stage-1 checkpoint specified for this run.')


def load_frozen_sae(trainer, checkpoint, device):
    ck=torch_load(checkpoint)
    if 'model' not in ck:raise RuntimeError('Expected Stage-1 checkpoint with model/args/step keys')
    a=ck.get('args',{})
    if a.get('vision_backend','dino')!='dino':raise RuntimeError('Only DINO checkpoint is supported.')
    cfg={k: a.get(k,default) for k,default in [
        ('vision_dim',1024),('canonical_dim',4096),('latent_dim',16384),
        ('bridge_bottleneck',2048),('align_dim',1024),('init_threshold',2.0)]}
    sae=trainer.SPAREBEAM(**cfg)
    sae.load_state_dict(ck['model'],strict=True)
    del ck
    trainer.freeze_module(sae)
    sae.to(device)
    return sae,cfg


def load_frozen_models(vision_path, llm_path, device, gc=False):
    from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(llm_path,use_fast=False)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    llm=AutoModelForCausalLM.from_pretrained(llm_path,torch_dtype=torch.bfloat16,
                                               low_cpu_mem_usage=True)
    llm.config.use_cache=not gc
    llm.eval()
    for p in llm.parameters():p.requires_grad_(False)
    if gc:
        try:llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        except TypeError:llm.gradient_checkpointing_enable()
    llm.to(device)
    processor=AutoImageProcessor.from_pretrained(vision_path)
    vision=AutoModel.from_pretrained(vision_path,torch_dtype=torch.float32)
    vision.eval()
    for p in vision.parameters():p.requires_grad_(False)
    vision.to(device)
    return vision,llm,tok,processor


