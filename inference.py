from argparse import ArgumentParser

from tokenizers import Tokenizer
import torch

import config as C
from sampler import Sampler
from lora_modules import get_model_from_config
from utils import modify_config


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--lora-checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default=C.DEVICE)
    parser.add_argument('--tokenizer', type=str, default=C.TOKENIZER_PATH)
    parser.add_argument('--temparature', type=float, default=1.)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--seed', type=str, required=True)
    parser.add_argument('--maxlen', type=int, default=C.MAXLEN)

    args = parser.parse_args()
    
    checkpoint = torch.load(args.checkpoint, args.device)
    lora_checkpoint = torch.load(args.lora_checkpoint, args.device)
    
    settings = checkpoint['settings']
    modify_config(C, **settings)
    
    model = get_model_from_config(settings)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.load_state_dict(lora_checkpoint['model'], strict=False)
    
    tokenizer = Tokenizer.from_file(args.tokenizer)

    sampler = Sampler(model, tokenizer, args.device, args.temparature)

    seed = sampler.sample(args.seed, args.topk, args.maxlen)
    print(seed)
