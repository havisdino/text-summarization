import torch
from torch import nn
import os
import loralib as lora
import config as C
from lora_modules import get_model_config


class Saver:
    def __init__(self, checkpoint_retention, checkpoint_interval):
        self.checkpoint_retention = checkpoint_retention * checkpoint_interval
        
    def _build_checkpoint(self, model):
        if isinstance(model, nn.DataParallel):
            model_state_dict = lora.lora_state_dict(model.module)
        elif isinstance(model, nn.Module):
            model_state_dict = lora.lora_state_dict(model)
        
        self.last_checkpoint = dict(
            model=model_state_dict,
            settings=get_model_config()
        )
    
    def save(self, model, step, label='lora'):
        if not os.path.exists('./checkpoints'):
            os.makedirs('checkpoints')
            
        path = (f'checkpoints/{label}-{C.ARCHITECTURE}-D{C.D_MODEL}'
                + f'-H{C.N_HEADS}-B{C.N_BLOCKS}-{step}.pt')
        
        last_kth = (f'checkpoints/{label}-{C.ARCHITECTURE}-D{C.D_MODEL}'
                    + f'-H{C.N_HEADS}-B{C.N_BLOCKS}-{step - self.checkpoint_retention}.pt')
        
        if os.path.exists(last_kth):
            os.remove(last_kth)
            
        self._build_checkpoint(model)
        torch.save(self.last_checkpoint, path)
