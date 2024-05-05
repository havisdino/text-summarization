# ---------- Transformer settings ----------
D_MODEL = 768
DFF = D_MODEL * 4
N_HEADS = 12
N_BLOCKS = 11
MAXLEN = 512
VOCAB_SIZE = 30000
DROPOUT = 0.1
WEIGHT_STD = 0.05
ARCHITECTURE = 'rezero' # options: 'rezero', 'vanilla'
# ------------------------------------------------------------


# ---------- Training settings ----------
LEARNING_RATE = 1e-5
N_STEPS = 1000
BATCH_SIZE = 16
GRAD_ACCUM_STEP = 100
DEVICE = 'cuda'
CHECKPOINT_STEP = 10     # Save the model after <CHECKPOINT_STEP> steps of grad accumulation
PREFETCH_FACTOR = 2
USE_AMP = True
TRAIN_LIMIT = None
VAL_LIMIT = 100 * BATCH_SIZE     # number of samples
CHECKPOINT_RETENTION = 2
# ------------------------------------------------------------


# ---------- Data settings ----------
END_TOKEN_ID = 29998
SUM_TOKEN_ID = 29999
TOKENIZER_PATH = 'tokenizer/byte-level-bpe-tinystories.json'
# ------------------------------------------------------------
