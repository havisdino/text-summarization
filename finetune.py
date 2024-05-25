from argparse import ArgumentParser
from tokenizers import Tokenizer
import torch
from dataset import CSVTextDataset
from lora_modules import get_model_from_config
import config as C
from trainer import Trainer
from utils import count_params, modify_config
from dataset import TokenDataset, collate_fn
from torch.utils.data import DataLoader
from torch import nn
import loralib as lora


parser = ArgumentParser()
parser.add_argument('--from-checkpoint', type=str, required=True)
parser.add_argument('--traindata', type=str, required=True)
parser.add_argument('--valdata', type=str, default=None)
parser.add_argument('--data-parallel', type=bool, default=False)

args = parser.parse_args()

tokenizer = Tokenizer.from_file(C.TOKENIZER_PATH)

checkpoint = torch.load(args.from_checkpoint, C.DEVICE)
settings = checkpoint['settings']
modify_config(C, **settings)

model = get_model_from_config(settings) 
model.load_state_dict(checkpoint['model'], strict=False)
print('Checkpoint loaded, default settings might be ignored')
start_step = 0

lora.mark_only_lora_as_trainable(model)
count_params(model)

if args.data_parallel:
    model = nn.DataParallel(model)
model.to(C.DEVICE)

scaler = torch.cuda.amp.GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=C.LEARNING_RATE)

scaler.load_state_dict(checkpoint['scaler'])
# optimizer.load_state_dict(checkpoint['optimizer'])

if args.traindata.endswith('.csv'):
    traindata = CSVTextDataset(args.traindata, C.MAXLEN + 1, tokenizer, limit=C.TRAIN_LIMIT, n_overlap=C.N_OVERLAP)
    if args.valdata is not None:
        valdata = CSVTextDataset(args.valdata, C.MAXLEN + 1, tokenizer, limit=C.VAL_LIMIT)
elif args.traindata.endswith('.bds'):
    traindata = TokenDataset(args.traindata, C.MAXLEN + 1, C.MAXLEN // 4, limit=C.TRAIN_LIMIT)
    if args.valdata is not None:
        valdata = TokenDataset(args.valdata, C.MAXLEN + 1, 0, limit=C.VAL_LIMIT)

loader_settings = dict(
    batch_size=C.BATCH_SIZE,
    collate_fn=collate_fn,
    prefetch_factor=C.PREFETCH_FACTOR,
    num_workers=2,
    drop_last=True
)
trainloader = DataLoader(traindata, **loader_settings)
if args.valdata is not None:
    valloader = DataLoader(valdata, **loader_settings)
else:
    valloader = None

trainer = Trainer(
    model, optimizer, scaler,
    C.VOCAB_SIZE, C.USE_AMP, C.DEVICE,
    C.GRAD_ACCUM_STEP, C.CHECKPOINT_RETENTION,
    C.CHECKPOINT_STEP, start_step
)

trainer.fit(trainloader, valloader, C.N_STEPS)
