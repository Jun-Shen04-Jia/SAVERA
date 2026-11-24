#!/usr/bin/env python3
"""CC3M caption data and prompt/answer collation for PRIVA training."""
import random
import torch
from torch.utils.data import Dataset


class CaptionWrapper(Dataset):
    def __init__(self,cc3m):self.cc3m=cc3m
    def __len__(self):return len(self.cc3m)
    def __getitem__(self,i):
        path,caption=self.cc3m[i]
        return {'image_path':str(path),'question':'','answer':str(caption),'task':'caption'}


class MixedIndex(Dataset):
    def __init__(self,sources,pairs):self.sources=sources;self.pairs=pairs
    def __len__(self):return len(self.pairs)
    def __getitem__(self,i):
        source,idx=self.pairs[i]
        return self.sources[source][idx]


def split_sources(sources,seed,val_fraction):
    train=[];val=[]
    for src,ds in enumerate(sources):
        idx=list(range(len(ds)));random.Random(seed+src*381).shuffle(idx)
        if len(idx)<2:raise ValueError('Each dataset needs at least two entries for validation')
        nval=max(1,int(len(idx)*val_fraction))
        val.extend((src,k) for k in idx[:nval])
        train.extend((src,k) for k in idx[nval:])
    return train,val


class PromptAnswerCollator:
    def __init__(self,image_collator,tokenizer,max_prompt=96,max_answer=64,
                 caption_prompt='Describe the image accurately and briefly.\nAnswer:'):
        self.img=image_collator;self.tok=tokenizer
        self.max_prompt=max_prompt;self.max_answer=max_answer
        self.caption_prompt=caption_prompt
        self.bos=tokenizer.bos_token_id
        self.eos=tokenizer.eos_token_id
        self.pad=tokenizer.pad_token_id
        if self.bos is None or self.eos is None or self.pad is None:
            raise ValueError('Tokenizer needs BOS, EOS, PAD')

    def _pad(self,arrays,pad_value):
        maxlen=max(len(a) for a in arrays)
        return torch.tensor([a+[pad_value]*(maxlen-len(a)) for a in arrays],dtype=torch.long)

    def tokenize_prompt(self,prompt):
        return [self.bos]+self.tok.encode(prompt,add_special_tokens=False)[:self.max_prompt-1]

    def __call__(self,items):
        prompts=[];fulls=[];labels=[];teachers=[];imgs=[];tasks=[]
        for x in items:
            prompt=self.caption_prompt
            ans=x['answer']
            prompt_ids=self.tokenize_prompt(prompt)
            ans_ids=self.tok.encode(ans,add_special_tokens=False)[:self.max_answer-1]+[self.eos]
            full=prompt_ids+ans_ids
            # First answer token is supervised, prefix and prompt tokens are not.
            gold=[-100]*len(prompt_ids)+ans_ids
            prompts.append(prompt_ids);fulls.append(full);labels.append(gold)
            teachers.append([self.bos]+ans_ids)
            imgs.append(self.img._image_tensor(x['image_path']))
            tasks.append(x['task'])
        pp=self._pad(prompts,self.pad)
        aa=self._pad(teachers,self.pad)
        ff=self._pad(fulls,self.pad)
        # Pad token may intentionally equal EOS for Vicuna. Make masks from lengths,
        # NEVER `tokens != pad_token_id` (would mask real EOS tokens).
        def mask(seq):
            return torch.tensor([[int(j<len(x)) for j in range(max(map(len,seq)))]
                                 for x in seq],dtype=torch.long)
        return {'pixel_values':torch.stack(imgs,0),
            'prompt_ids':pp,'prompt_mask':mask(prompts),
            'teacher_ids':aa,'teacher_mask':mask(teachers),
            'input_ids':ff,'attention_mask':mask(fulls),
            'labels':self._pad(labels,-100),'tasks':tasks}
