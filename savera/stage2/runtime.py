#!/usr/bin/env python3
"""Common frozen-model loaders and prompt-only Text SAE extraction."""
import torch
import torch.nn.functional as F
from ..common.model_loaders import (file_fingerprint,torch_load,
                            load_frozen_sae,load_frozen_models,
                            confirm_sae_fingerprint)
from .model import PRIVA
from ..common.cuda_utils import sync_stage, enabled


def frozen_visuals(trainer,vision,sae,pixels,temperature,vision_layer,with_hv=False):
    # One DINO tower forward only, including optional visual reconstruction target.
    with torch.no_grad():
        sync_stage('frozen_visuals: start DINO')
        with torch.autocast(device_type='cuda',enabled=False) if pixels.is_cuda else torch.no_grad():
            h=trainer.extract_vision_tokens(vision,'dino',pixels.float(),vision_layer)
        sync_stage('frozen_visuals: finished DINO')
        if h.shape[1:]!=(576,1024):
            raise RuntimeError('Expected DINO patch feature (576,1024), got '+str(h.shape))
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not enabled('SAVERA_SAE_FP32')) if pixels.is_cuda else torch.no_grad():
            vz=sae.encode_vision(h,temperature)['z']
            sync_stage('frozen_visuals: finished Vision SAE')
            bz=sae.encode_bridge(vz,temperature)
            sync_stage('frozen_visuals: finished Bridge SAE')
    if with_hv:return vz.detach(),bz['z'].detach(),bz['x_prime'].detach(),h.detach()
    return vz.detach(),bz['z'].detach(),bz['x_prime'].detach()


def frozen_text_sae(trainer,sae,llm,ids,mask,text_layer,temperature,teacher=False):
    """Only prompt tokens for inputs; answer tokens may be used as stop-grad teacher."""
    with torch.no_grad():
        sync_stage('frozen_text_sae: starting Vicuna text hidden (teacher=%s)'%teacher)
        states=trainer.extract_text_hidden(llm,ids,mask,text_layer)
        sync_stage('frozen_text_sae: finished Vicuna text hidden (teacher=%s)'%teacher)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not enabled('SAVERA_SAE_FP32')) if ids.is_cuda else torch.no_grad():
            zs=sae.encode_text(states.float() if enabled('SAVERA_SAE_FP32') else states,temperature)['z']
            if teacher:
                pooled=(zs.float()*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
                teacher_out=F.normalize(sae.align_head(pooled.to(sae.align_head.weight.dtype)).float(),dim=-1).detach()
                sync_stage('frozen_text_sae: finished teacher Text SAE')
                return teacher_out
    sync_stage('frozen_text_sae: finished Text SAE (teacher=%s)'%teacher)
    return zs.detach()


def answer_masked_loss(llm,prefix,input_ids,attention_mask,labels):
    embed=llm.get_input_embeddings()(input_ids)
    prefix=prefix.to(embed.dtype)
    inp=torch.cat([prefix,embed],dim=1)
    vis_mask=torch.ones(prefix.shape[:2],device=prefix.device,dtype=attention_mask.dtype)
    mask=torch.cat([vis_mask,attention_mask],dim=1)
    ign=torch.full(prefix.shape[:2],-100,device=prefix.device,dtype=labels.dtype)
    sync_stage('answer_masked_loss: entering Vicuna with gradient inputs')
    loss=llm(inputs_embeds=inp,attention_mask=mask,
             labels=torch.cat([ign,labels],dim=1),use_cache=False,return_dict=True).loss
    sync_stage('answer_masked_loss: Vicuna forward completed')
    return loss


