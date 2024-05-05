import torch
from torch import nn
from config import *


def get_causal_mask(maxlen):
    causal_mask = torch.ones(maxlen, maxlen).tril() == 0
    causal_mask = torch.where(causal_mask, -float('inf'), 0)
    return causal_mask


def init_weights(m):
    for p in m.parameters():
        nn.init.normal_(p, std=WEIGHT_STD)


def count_params(model):
    if isinstance(model, nn.DataParallel):
        n_params = sum(p.numel() for p in model.module.parameters())
    elif isinstance(model, nn.Module):
        n_params = sum(p.numel() for p in model.parameters())
        
    print(f'Parameters: {n_params:,}')
    return n_params
    

def set_description_bar(bar, step, n_steps, **kwargs):
    description = f'step {step}/{n_steps}'
    bar.set_description(description)
    bar.set_postfix(kwargs)
        
        
def write_tensorboard_logs(writer, *, global_step, loss=None, ppl=None, val_ppl=None, lr=None):
    if loss is not None:
        writer.add_scalar('loss/train', loss, global_step)
        writer.add_scalar('ppl/train', ppl, global_step)
    
    if val_ppl is not None:    
        writer.add_scalar('ppl/val', val_ppl, global_step)
    
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
        

def modify_config(config, **kwargs):
    for key, item in kwargs.items():
        setattr(config, key.upper(), item)   
