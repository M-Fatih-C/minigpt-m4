import os


def env_int(name, default):
    return int(
        os.getenv(name, default)
    )


def env_float(name, default):
    return float(
        os.getenv(name, default)
    )


# --------------------------------------------------
# Dataset / tokenizer
# --------------------------------------------------

VOCAB_SIZE = env_int(
    "MINIGPT_VOCAB_SIZE",
    2048,
)

TRAIN_DATA_PATH = "data/train.bin"
VALID_DATA_PATH = "data/validation.bin"


# --------------------------------------------------
# Sequence
# --------------------------------------------------

BLOCK_SIZE = env_int(
    "MINIGPT_BLOCK_SIZE",
    256,
)

BATCH_SIZE = env_int(
    "MINIGPT_BATCH_SIZE",
    32,
)


# --------------------------------------------------
# Transformer
# --------------------------------------------------

N_EMBD = env_int(
    "MINIGPT_N_EMBD",
    384,
)

N_HEAD = env_int(
    "MINIGPT_N_HEAD",
    6,
)

N_LAYER = env_int(
    "MINIGPT_N_LAYER",
    8,
)

if N_EMBD % N_HEAD != 0:
    raise ValueError(
        "N_EMBD must be divisible by N_HEAD"
    )

HEAD_SIZE = (
    N_EMBD // N_HEAD
)

DROPOUT = env_float(
    "MINIGPT_DROPOUT",
    0.1,
)


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
