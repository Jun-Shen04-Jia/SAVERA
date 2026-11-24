#!/usr/bin/env python3
"""Optional CUDA diagnostics for SAVERA training."""
import os
import torch


def enabled(name):
    return os.environ.get(name, '0').lower() in ('1','true','yes','on')


def configure_cuda():
    if not torch.cuda.is_available():
        print('[CUDA_DIAG] CUDA unavailable',flush=True)
        return
    if enabled('SAVERA_FORCE_MATH_SDPA'):
        backends=torch.backends.cuda
        backends.enable_flash_sdp(False)
        backends.enable_mem_efficient_sdp(False)
        if hasattr(backends,'enable_cudnn_sdp'):
            backends.enable_cudnn_sdp(False)
        backends.enable_math_sdp(True)
        print('[CUDA_DIAG] forced MATH SDPA; flash/mem-efficient/cudnn SDPA disabled',flush=True)
    if enabled('SAVERA_CUDA_DEBUG'):
        props=torch.cuda.get_device_properties(0)
        print('[CUDA_DIAG] torch',torch.__version__,'torch CUDA',torch.version.cuda,
              'device',torch.cuda.get_device_name(0),'compute capability',
              '%d.%d'%(props.major,props.minor),flush=True)


def sync_stage(name):
    if not enabled('SAVERA_CUDA_DEBUG'):return
    # Synchronizing at each stage makes asynchronous CUDA faults much easier to localize.
    torch.cuda.synchronize()
    a=torch.cuda.memory_allocated()/2**30
    r=torch.cuda.memory_reserved()/2**30
    print('[CUDA_STAGE]',name,'allocated_GiB=%.3f'%a,'reserved_GiB=%.3f'%r,flush=True)


