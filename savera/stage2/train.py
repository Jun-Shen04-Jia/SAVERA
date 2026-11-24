#!/usr/bin/env python3
"""Stage 2: PRIVA training with frozen SPARE--BEAM, DINOv2, and Vicuna."""
import argparse
import json
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..common.cuda_utils import configure_cuda, sync_stage, enabled
from .model import (PRIVA,relative_delta_regularizer,
                             visual_preserve_loss,semantic_align_loss)
from ..common.model_loaders import (file_fingerprint, torch_load, load_frozen_sae,
                           load_frozen_models, confirm_sae_fingerprint)
from .runtime import (frozen_visuals, frozen_text_sae, answer_masked_loss)
from .data import CaptionWrapper,MixedIndex,split_sources,PromptAnswerCollator
from ..stage1 import train as trainer


def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('spare_beam_ckpt','vision_model','text_model','manifest','output_dir'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--image_root',default='')
    p.add_argument('--max_samples',type=int,default=20000)
    p.add_argument('--val_fraction',type=float,default=.03)
    p.add_argument('--batch_size',type=int,default=2)
    p.add_argument('--grad_accum',type=int,default=4)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--max_steps',type=int,default=2500)
    p.add_argument('--num_workers',type=int,default=4)
    p.add_argument('--mode',choices=['vbt','vb','v_only','b_only'],default='vbt')
    p.add_argument('--d_model',type=int,default=512)
    p.add_argument('--heads',type=int,default=8)
    p.add_argument('--max_prompt',type=int,default=96)
    p.add_argument('--max_answer',type=int,default=64)
    p.add_argument('--caption_prompt',default='Describe the image accurately and briefly.\\nAnswer:')
    p.add_argument('--sae_temperature',type=float,default=.10)
    p.add_argument('--vision_layer',type=int,default=-2)
    p.add_argument('--text_layer',type=int,default=16)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--weight_decay',type=float,default=.01)
    p.add_argument('--lambda_align',type=float,default=.05)
    p.add_argument('--lambda_visual',type=float,default=.05)
    p.add_argument('--lambda_delta',type=float,default=.02)
    p.add_argument('--max_grad_norm',type=float,default=1.0)
    p.add_argument('--save_every',type=int,default=200)
    p.add_argument('--val_every',type=int,default=200)
    p.add_argument('--val_batches',type=int,default=16)
    p.add_argument('--log_every',type=int,default=20)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',default='')
    return p.parse_args()


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def make_loader(sources,indices,collator,args,epoch=0,shuffle=False):
    g=torch.Generator().manual_seed(args.seed+epoch)
    return DataLoader(MixedIndex(sources,indices),batch_size=args.batch_size,
                      shuffle=shuffle,generator=g,num_workers=args.num_workers,
                      pin_memory=True,collate_fn=collator,
                      persistent_workers=args.num_workers>0)


def as_gpu(batch,device):
    return {k:(v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v)
            for k,v in batch.items()}


def forward_loss(net,trainer,vision,sae,llm,batch,args,teacher=True):
    sync_stage('forward_loss: start')
    zv,zb,base,hv=frozen_visuals(trainer,vision,sae,batch['pixel_values'].float(),
                                 args.sae_temperature,args.vision_layer,with_hv=True)
    with torch.no_grad():
        if args.mode=='vbt':
            zt=frozen_text_sae(trainer,sae,llm,batch['prompt_ids'],batch['prompt_mask'],
                               args.text_layer,args.sae_temperature)
        else:zt=None
        if teacher and args.lambda_align>0:
            sem_teacher=frozen_text_sae(trainer,sae,llm,batch['teacher_ids'],
                      batch['teacher_mask'],args.text_layer,args.sae_temperature,teacher=True)
        else:sem_teacher=None
    with torch.autocast('cuda',dtype=torch.bfloat16):
        prefix,aux=net(zv,zb,base,zt,batch['prompt_mask'] if zt is not None else None,return_aux=True)
        sync_stage('forward_loss: fusion forward complete')
        lm=answer_masked_loss(llm,prefix,batch['input_ids'],batch['attention_mask'],batch['labels'])
        visual=visual_preserve_loss(aux['visual_pred'],hv) if args.lambda_visual>0 else lm.new_zeros(())
        align=semantic_align_loss(aux['semantic'],sem_teacher) if sem_teacher is not None else lm.new_zeros(())
        reg=relative_delta_regularizer(aux['delta'],base) if args.lambda_delta>0 else lm.new_zeros(())
        loss=lm+args.lambda_visual*visual+args.lambda_align*align+args.lambda_delta*reg
        sync_stage('forward_loss: total loss complete')
    return loss,{'lm':lm.detach(),'visual':visual.detach(),'align':align.detach(),
                 'delta':reg.detach(),'gate':aux['gate_mean'].detach(),
                 'prefix_gate':torch.sigmoid(net.gate_logit.detach())}


@torch.no_grad()
def validate(net,trainer,vision,sae,llm,loader,args,device):
    net.eval();llm.eval()
    totals=[]
    for i,batch in enumerate(loader):
        if i>=args.val_batches:break
        batch=as_gpu(batch,device)
        zv,zb,base,hv=frozen_visuals(trainer,vision,sae,batch['pixel_values'].float(),
                                     args.sae_temperature,args.vision_layer,with_hv=True)
        zt=(frozen_text_sae(trainer,sae,llm,batch['prompt_ids'],batch['prompt_mask'],
                      args.text_layer,args.sae_temperature) if args.mode=='vbt' else None)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            fused=net(zv,zb,base,zt,batch['prompt_mask'] if zt is not None else None)
            l_new=answer_masked_loss(llm,fused,batch['input_ids'],batch['attention_mask'],batch['labels'])
            l_base=answer_masked_loss(llm,base,batch['input_ids'],batch['attention_mask'],batch['labels'])
            # Stage-1 Bridge captioning uses caption tokens without an instruction prompt.
            l_stage1_caption=trainer.bridge_lm_loss(llm,base,batch['teacher_ids'],batch['teacher_mask'])
        totals.append((float(l_new),float(l_base),float(l_stage1_caption)))
    net.train();llm.train()
    return {'val_lm_fusion':sum(v[0] for v in totals)/len(totals) if totals else None,
            'val_lm_baseline':sum(v[1] for v in totals)/len(totals) if totals else None,
            'val_lm_stage1_caption':sum(v[2] for v in totals)/len(totals) if totals else None,
            'val_batches':len(totals)}


def public_args(args):
    """Keep local paths out of saved checkpoints and metadata."""
    hidden={'vision_model','text_model','manifest','image_root','output_dir','spare_beam_ckpt','resume'}
    return {k: ('<local-path>' if k in hidden and v else v) for k,v in vars(args).items()}


def safe_save(path,net,opt,sched,args,step,epoch,ib,config,source,history,split_sizes):
    rec={'fusion':net.state_dict(),'optimizer':opt.state_dict(),
         'scheduler':sched.state_dict(),'fusion_config':config,
         'args':public_args(args),'sae_fingerprint':source,'global_step':step,
         'epoch':epoch,'batch_in_epoch':ib,'history':history,'split_sizes':split_sizes}
    tmp=path.with_suffix('.writing.pt');torch.save(rec,tmp);os.replace(tmp,path)


def main():
    a=args_parser()
    if not torch.cuda.is_available():raise RuntimeError('Training requires an NVIDIA GPU')
    if not 0<a.val_fraction<0.5:raise ValueError('val_fraction out of range')
    if a.grad_accum<1:raise ValueError('grad_accum must be >=1')
    seed_all(a.seed)
    configure_cuda()
    device=torch.device('cuda');torch.backends.cuda.matmul.allow_tf32=True
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    sae_fp=file_fingerprint(a.spare_beam_ckpt)
    sae,cfg=load_frozen_sae(trainer,a.spare_beam_ckpt,device)
    if cfg['vision_dim']!=1024 or cfg['canonical_dim']!=4096:
        raise RuntimeError('Expected DINO 1024 and Vicuna 4096')
    vision,llm,tok,processor=load_frozen_models(a.vision_model,a.text_model,device,gc=not enabled('SAVERA_DISABLE_GC'))
    llm.config.use_cache=False
    if not enabled('SAVERA_DISABLE_GC'):
        llm.train() # Enable gradient checkpointing for the frozen language backbone.
    else:
        llm.eval() # Frozen LLM still propagates gradient to visual prefix in eval mode.
    sources=[CaptionWrapper(trainer.CC3MPairedDataset(a.manifest,a.image_root,a.max_samples))]
    train_idx,val_idx=split_sources(sources,a.seed,a.val_fraction)
    stage1_collator=trainer.CC3MCollator(tok,getattr(processor,'image_mean',[.485,.456,.406]),
                            getattr(processor,'image_std',[.229,.224,.225]),
                            image_size=336,max_text_len=a.max_answer,pad_square=True)
    c=PromptAnswerCollator(stage1_collator,tok,a.max_prompt,a.max_answer,
                           a.caption_prompt.replace('\\n','\n'))
    v_dl=make_loader(sources,val_idx,c,a,shuffle=False)
    config=dict(latent_dim=cfg['latent_dim'],d_model=a.d_model,llm_dim=cfg['canonical_dim'],
                vision_dim=cfg['vision_dim'],align_dim=cfg['align_dim'],heads=a.heads,mode=a.mode)
    net=PRIVA(**config).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    batches_per_epoch=math.ceil(len(train_idx)/a.batch_size)
    steps=max(1,a.max_steps if a.max_steps>0 else a.epochs*math.ceil(batches_per_epoch/a.grad_accum))
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda st:0.5*(1+math.cos(math.pi*min(st,steps)/steps)))
    step=0;epoch_start=0;batch_start=0;history=[]
    if a.resume:
        r=torch_load(a.resume);confirm_sae_fingerprint(r.get('sae_fingerprint'),sae_fp)
        if r['fusion_config']!=config:raise ValueError('Resume architecture mismatch')
        saved_args=r.get('args',{})
        for k in ['max_samples','val_fraction','batch_size','grad_accum','seed','caption_prompt','max_answer','max_prompt','mode','max_steps','lr','lambda_align','lambda_visual','lambda_delta','sae_temperature','vision_layer','text_layer']:
            if saved_args.get(k)!=getattr(a,k):raise ValueError('Resume dataset/training argument mismatch: '+k)
        net.load_state_dict(r['fusion']);opt.load_state_dict(r['optimizer'])
        sch.load_state_dict(r['scheduler']);step=int(r['global_step'])
        epoch_start=int(r['epoch']);batch_start=int(r['batch_in_epoch']);history=r.get('history',[])
        print('RESUME',a.resume,step,flush=True)
    (out/'config.json').write_text(json.dumps({'args':public_args(a),'spare_beam_checkpoint':sae_fp,'config':config,
                  'training_items':len(train_idx),'validation_items':len(val_idx)},indent=2))
    print('Frozen SPARE--BEAM checkpoint',sae_fp,'train',len(train_idx),'val',len(val_idx),
          'PRIVA parameters',sum(v.numel() for v in net.parameters()),flush=True)
    if step>=steps:print('Already finished requested steps');return
    opt.zero_grad(set_to_none=True)
    running=[];end_epoch=epoch_start;end_batch=batch_start
    for epoch in range(epoch_start,a.epochs):
        train_dl=make_loader(sources,train_idx,c,a,epoch,shuffle=True)
        accum=0
        for i,batch in enumerate(train_dl):
            if epoch==epoch_start and i<batch_start:continue
            if step>=steps:break
            batch=as_gpu(batch,device)
            sync_stage('train: batch uploaded')
            block_start=(i//a.grad_accum)*a.grad_accum
            denom=min(a.grad_accum,len(train_dl)-block_start)
            loss,metrics=forward_loss(net,trainer,vision,sae,llm,batch,a)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            sync_stage('train: before backward')
            (loss/denom).backward()
            sync_stage('train: backward finished')
            accum+=1
            running.append({k:float(v) for k,v in metrics.items()})
            if accum>=a.grad_accum or i==len(train_dl)-1:
                torch.nn.utils.clip_grad_norm_(net.parameters(),a.max_grad_norm)
                opt.step();sync_stage('train: optimizer step finished')
                sch.step();opt.zero_grad(set_to_none=True)
                accum=0;step+=1;end_epoch=epoch;end_batch=i+1
                if step%a.log_every==0:
                    r={k:sum(x[k] for x in running)/len(running) for k in running[0]}
                    r.update(step=step,epoch=epoch,lr=opt.param_groups[0]['lr'])
                    print('TRAIN',json.dumps(r),flush=True);running=[]
                if a.val_every>0 and step%a.val_every==0:
                    scores=validate(net,trainer,vision,sae,llm,v_dl,a,device)
                    scores['step']=step;history.append(scores)
                    with (out/'val_history.jsonl').open('a') as fp:fp.write(json.dumps(scores)+'\n')
                    print('VALID',json.dumps(scores),flush=True)
                if step%a.save_every==0:
                    safe_save(out/'latest_priva.pt',net,opt,sch,a,step,epoch,i+1,config,sae_fp,history,(len(train_idx),len(val_idx)))
                    print('SAVE',step,flush=True)
            if step>=steps:break
        if step>=steps:break
    safe_save(out/'latest_priva.pt',net,opt,sch,a,step,end_epoch,end_batch,config,sae_fp,history,(len(train_idx),len(val_idx)))
    print('COMPLETE',out/'latest_priva.pt','steps',step,flush=True)

if __name__=='__main__':main()
