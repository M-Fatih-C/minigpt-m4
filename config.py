# MiniGPT-M4 configuration

# --------------------------------------------------
# Dataset / tokenizer
# --------------------------------------------------

VOCAB_SIZE = 2048

TRAIN_DATA_PATH = "data/train.bin"
VALID_DATA_PATH = "data/validation.bin"


# --------------------------------------------------
# Sequence
# --------------------------------------------------

BLOCK_SIZE = 256
BATCH_SIZE = 32


# --------------------------------------------------
# Transformer
# --------------------------------------------------

N_EMBD = 384

N_HEAD = 6

N_LAYER = 8

assert N_EMBD % N_HEAD == 0

HEAD_SIZE = N_EMBD // N_HEAD

DROPOUT = 0.1

# --------------------------------------------------
# Training
# --------------------------------------------------

MAX_STEPS = 5000

LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5

WARMUP_STEPS = 200

WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

EVAL_INTERVAL = 250
EVAL_ITERS = 50

CHECKPOINT_INTERVAL = 500
SAMPLE_INTERVAL = 500

SAMPLE_TOKENS = 200

SEED = 42