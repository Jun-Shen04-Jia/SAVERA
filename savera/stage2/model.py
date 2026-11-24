#!/usr/bin/env python3
"""Stage 2: PRIVA prompt-conditioned residual integration of Vision, Bridge, and Text."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class PRIVA(nn.Module):
    def __init__(self, latent_dim=16384, d_model=512, llm_dim=4096, vision_dim=1024,
                 align_dim=1024, heads=8, dropout=0.0, mode='vbt'):
        super().__init__()
        if mode not in ('vbt','vb','v_only','b_only'):
            raise ValueError('mode: vbt, vb, v_only, or b_only')
        if d_model % heads or d_model % 4:
            raise ValueError('d_model must be divisible by heads and 4')
        self.cfg = dict(latent_dim=latent_dim,d_model=d_model,llm_dim=llm_dim,
                        vision_dim=vision_dim,align_dim=align_dim,heads=heads,
                        dropout=dropout,mode=mode)
        self.mode=mode
        self.norm_v=nn.LayerNorm(latent_dim)
        self.norm_b=nn.LayerNorm(latent_dim)
        self.norm_t=nn.LayerNorm(latent_dim)
        self.proj_v=nn.Linear(latent_dim,d_model)
        self.proj_b=nn.Linear(latent_dim,d_model)
        self.proj_t=nn.Linear(latent_dim,d_model)
        # Sigmoid is NOT an attention probability over modalities: each feature can
        # preserve Vision and incorporate complementary Bridge information.
        self.vb_gate=nn.Sequential(nn.Linear(d_model*3,d_model),nn.SiLU(),
                                   nn.Linear(d_model,d_model))
        self.patch_norm=nn.LayerNorm(d_model)
        self.spatial_attn=nn.MultiheadAttention(d_model,heads,dropout=dropout,batch_first=True)
        self.spatial_norm=nn.LayerNorm(d_model)
        self.text_attn=nn.MultiheadAttention(d_model,heads,dropout=dropout,batch_first=True)
        self.text_norm=nn.LayerNorm(d_model)
        self.ffn=nn.Sequential(nn.Linear(d_model,d_model*4),nn.GELU(),nn.Dropout(dropout),
                               nn.Linear(d_model*4,d_model))
        self.final_norm=nn.LayerNorm(d_model)
        self.out_proj=nn.Linear(d_model,llm_dim)
        self.visual_probe=nn.Linear(d_model,vision_dim)
        self.semantic_head=nn.Linear(d_model,align_dim)
        self.gate_logit=nn.Parameter(torch.tensor(-3.0))
        self.pos_scale=nn.Parameter(torch.tensor(0.10))
        nn.init.normal_(self.out_proj.weight,std=1e-3)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.vb_gate[-1].bias)

    @staticmethod
    def position(n,d,device,dtype):
        side=math.isqrt(n)
        if side*side!=n:
            raise ValueError('Expected square DINO patch grid, e.g. 576=24x24')
        y,x=torch.meshgrid(torch.arange(side,device=device),torch.arange(side,device=device),indexing='ij')
        scale=torch.exp(-math.log(10000.)*torch.arange(d//4,device=device,dtype=torch.float32)/max(1,d//4-1))
        y=y.reshape(-1,1).float()*scale
        x=x.reshape(-1,1).float()*scale
        return torch.cat([y.sin(),y.cos(),x.sin(),x.cos()],dim=-1)[None].to(dtype)

    def forward(self,z_v,z_b,bridge_prefix,z_t=None,text_mask=None,return_aux=False):
        if z_v.shape!=z_b.shape or z_v.ndim!=3 or z_v.shape[-1]!=self.cfg['latent_dim']:
            raise ValueError('z_v and z_b must be [B, patches, latent_dim]')
        if bridge_prefix.shape!=z_v.shape[:2]+(self.cfg['llm_dim'],):
            raise ValueError('Bridge decoder prefix shape mismatch')
        # Preserve individual (non-identical) modality representations.
        v=self.proj_v(self.norm_v(z_v))
        b=self.proj_b(self.norm_b(z_b))
        if self.mode=='v_only': b=torch.zeros_like(b)
        if self.mode=='b_only': v=torch.zeros_like(v)
        if self.mode=='vbt':
            if z_t is None or text_mask is None or z_t.ndim!=3:
                raise ValueError('vbt requires prompt-only z_t and text_mask')
            if z_t.shape[0]!=z_v.shape[0] or z_t.shape[-1]!=z_v.shape[-1]:
                raise ValueError('Text SAE dimension mismatch')
            t=self.proj_t(self.norm_t(z_t))
            mask=text_mask.bool()
            tpool=(t*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        else:
            t=None
            tpool=torch.zeros_like(v[:,0])
        tg=tpool.unsqueeze(1).expand(-1,v.shape[1],-1)
        gate=torch.sigmoid(self.vb_gate(torch.cat([v,b,tg],dim=-1)))
        if self.mode=='v_only':gate=torch.zeros_like(gate)
        if self.mode=='b_only':gate=torch.ones_like(gate)
        h=self.patch_norm(v+gate*b)
        pos=self.position(z_v.shape[1],h.shape[-1],h.device,h.dtype)*self.pos_scale
        h=h+pos
        spatial,_=self.spatial_attn(h,h,h,need_weights=False)
        h=self.spatial_norm(h+spatial)
        # Preserve patch-level features before global text-conditioned mixing.
        visual_repr=h
        if self.mode=='vbt':
            ctx,_=self.text_attn(h,t,t,key_padding_mask=~mask,need_weights=False)
            h=self.text_norm(h+ctx)
        h=self.final_norm(h+self.ffn(h))
        delta=self.out_proj(h)*torch.sigmoid(self.gate_logit)
        fused=bridge_prefix+delta.to(bridge_prefix.dtype)
        if return_aux:
            return fused, {'delta':delta,'gate_mean':gate.detach().mean(),
                           'visual_pred':self.visual_probe(visual_repr),
                           'semantic':F.normalize(self.semantic_head(h.float().mean(1)),dim=-1)}
        return fused


def relative_delta_regularizer(delta,base):
    return delta.float().square().mean()/base.detach().float().square().mean().clamp_min(1e-6)


def visual_preserve_loss(pred, original):
    pred=pred.float(); original=original.float().detach()
    # normalized reconstruction prevents DINO magnitude dominating other losses
    mse=(pred-original).square().mean()/original.square().mean().clamp_min(1e-6)
    cosine=(1-F.cosine_similarity(pred,original,dim=-1)).mean()
    return mse+0.2*cosine


def semantic_align_loss(sem,teacher):
    teacher=F.normalize(teacher.float().detach(),dim=-1)
    sem=F.normalize(sem.float(),dim=-1)
    cos=(1-(sem*teacher).sum(-1)).mean()
    if sem.size(0)<2:return cos
    logits=sem@teacher.t()/0.07
    labels=torch.arange(len(sem),device=sem.device)
    return cos+0.25*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.t(),labels))/2
