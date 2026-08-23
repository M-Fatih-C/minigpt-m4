import argparse
import math
import os
import random
import time

from pathlib import Path

import numpy as np
import torch

from tokenizers import Tokenizer

from config_v3 import (
    BATCH_SIZE,
    BLOCK_SIZE,
    VOCAB_SIZE,
)

from model_v3 import MiniGPTV3


# ============================================================
# MiniGPT-M4 V3 — 80M Pretraining
# ============================================================

SEED = 42


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

TRAIN_DATA_PATH = Path(
    "data/pretrain_v3/train.bin"
)

VALIDATION_DATA_PATH = Path(
    "data/pretrain_v3/validation.bin"
)

TOKENIZER_PATH = Path(
    "tokenizer_v3/tokenizer.json"
)


# ------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------

CHECKPOINT_DIR = Path(
    "checkpoints_v3"
)

LATEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "latest.pt"
)

BEST_MODEL = (
    CHECKPOINT_DIR
    / "best_model.pt"
)

FINAL_MODEL = (
    CHECKPOINT_DIR
    / "final_model.pt"
)


# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

PEAK_LR = 3e-4
MIN_LR = 3e-5

WARMUP_STEPS = 2_000

WEIGHT_DECAY = 0.1

BETA1 = 0.9
BETA2 = 0.95

GRAD_CLIP = 1.0


# ------------------------------------------------------------
# Runtime intervals
# ------------------------------------------------------------

LOG_INTERVAL = 25

EVAL_INTERVAL = 2_000
EVAL_BATCHES = 50

CHECKPOINT_INTERVAL = 1_000

SAMPLE_INTERVAL = 5_000

SAMPLE_NEW_TOKENS = 80


# ------------------------------------------------------------
# Pretraining samples
# ------------------------------------------------------------

SAMPLE_PROMPTS = [
    "Artificial intelligence is",
    "The history of computing",
    "A neural network learns",
]


# ============================================================
# Utility
# ============================================================

def set_seed(
    seed: int,
):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


def format_duration(
    seconds: float,
) -> str:

    seconds = max(
        0,
        int(seconds),
    )

    days, seconds = divmod(
        seconds,
        86400,
    )

    hours, seconds = divmod(
        seconds,
        3600,
    )

    minutes, seconds = divmod(
        seconds,
        60,
    )

    if days:

        return (
            f"{days}d "
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:02d}"
        )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def mps_memory():

    allocated = None
    driver = None

    if not hasattr(
        torch,
        "mps",
    ):
        return allocated, driver

    if hasattr(
        torch.mps,
        "current_allocated_memory",
    ):

        allocated = (
            torch.mps
            .current_allocated_memory()
            / 1024**3
        )

    if hasattr(
        torch.mps,
        "driver_allocated_memory",
    ):

        driver = (
            torch.mps
            .driver_allocated_memory()
            / 1024**3
        )

    return (
        allocated,
        driver,
    )


# ============================================================
# Dataset
# ============================================================

class BinaryTokenDataset:

    def __init__(
        self,
        path: Path,
    ):

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        self.path = path

        self.data = np.memmap(
            path,
            dtype="<u2",
            mode="r",
        )

        self.token_count = len(
            self.data
        )

        if (
            self.token_count
            <= BLOCK_SIZE
        ):

            raise RuntimeError(
                f"Dataset too small: "
                f"{path}"
            )

        # ----------------------------------------------------
        # Stride = BLOCK_SIZE
        #
        # Block 0:
        #
        # x = token 0..255
        # y = token 1..256
        #
        # Block 1:
        #
        # x = token 256..511
        # y = token 257..512
        #
        # Therefore every target token is trained
        # essentially once per epoch.
        # ----------------------------------------------------

        self.block_count = (
            (self.token_count - 1)
            // BLOCK_SIZE
        )

        self.offsets = np.arange(
            BLOCK_SIZE + 1,
            dtype=np.int64,
        )

    def get_batch(
        self,
        block_ids: np.ndarray,
        device: torch.device,
    ):

        block_ids = np.asarray(
            block_ids,
            dtype=np.int64,
        )

        starts = (
            block_ids
            * BLOCK_SIZE
        )

        indices = (
            starts[:, None]
            + self.offsets[None, :]
        )

        tokens = np.asarray(
            self.data[indices],
            dtype=np.int64,
        )

        x_numpy = np.ascontiguousarray(
            tokens[:, :-1]
        )

        y_numpy = np.ascontiguousarray(
            tokens[:, 1:]
        )

        x = (
            torch
            .from_numpy(x_numpy)
            .to(
                device=device,
                non_blocking=False,
            )
        )

        y = (
            torch
            .from_numpy(y_numpy)
            .to(
                device=device,
                non_blocking=False,
            )
        )

        return x, y


# ============================================================
# Learning rate
# ============================================================

def get_learning_rate(
    step: int,
    max_steps: int,
    warmup_steps: int,
):

    # --------------------------------------------------------
    # Linear warmup
    # --------------------------------------------------------

    if step < warmup_steps:

        return (
            PEAK_LR
            * (step + 1)
            / max(
                1,
                warmup_steps,
            )
        )

    # --------------------------------------------------------
    # Cosine decay
    # --------------------------------------------------------

    if step >= max_steps:

        return MIN_LR

    decay_steps = max(
        1,
        max_steps
        - warmup_steps,
    )

    progress = (
        step
        - warmup_steps
    ) / decay_steps

    progress = min(
        1.0,
        max(
            0.0,
            progress,
        ),
    )

    coefficient = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )

    return (
        MIN_LR
        + coefficient
        * (
            PEAK_LR
            - MIN_LR
        )
    )


# ============================================================
# Optimizer
# ============================================================

def build_optimizer(
    model,
):

    # --------------------------------------------------------
    # Weight decay:
    #
    # matrices     -> decay
    # RMSNorm etc. -> no decay
    # --------------------------------------------------------

    decay_params = []

    no_decay_params = []

    for parameter in model.parameters():

        if not parameter.requires_grad:
            continue

        if parameter.dim() >= 2:

            decay_params.append(
                parameter
            )

        else:

            no_decay_params.append(
                parameter
            )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    decay_params,

                "weight_decay":
                    WEIGHT_DECAY,
            },
            {
                "params":
                    no_decay_params,

                "weight_decay":
                    0.0,
            },
        ],
        lr=PEAK_LR,
        betas=(
            BETA1,
            BETA2,
        ),
    )

    decay_count = sum(
        p.numel()
        for p in decay_params
    )

    no_decay_count = sum(
        p.numel()
        for p in no_decay_params
    )

    print()
    print(
        f"Decay parameters    : "
        f"{decay_count:,}"
    )

    print(
        f"No-decay parameters : "
        f"{no_decay_count:,}"
    )

    return optimizer


def move_optimizer_to_device(
    optimizer,
    device,
):

    for state in (
        optimizer.state.values()
    ):

        for key, value in list(
            state.items()
        ):

            if torch.is_tensor(
                value
            ):

                state[key] = (
                    value.to(
                        device
                    )
                )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataset,
    device,
    batch_size,
    eval_batches,
):

    model.eval()

    rng = np.random.default_rng(
        SEED + 100_000
    )

    losses = []

    for _ in range(
        eval_batches
    ):

        block_ids = rng.integers(
            low=0,
            high=dataset.block_count,
            size=batch_size,
            dtype=np.int64,
        )

        x, y = dataset.get_batch(
            block_ids,
            device,
        )

        _, loss = model(
            x,
            y,
        )

        losses.append(
            loss.item()
        )

    model.train()

    return (
        sum(losses)
        / len(losses)
    )


# ============================================================
# Sampling
# ============================================================

@torch.no_grad()
def generate_samples(
    model,
    tokenizer,
    device,
    step,
):

    model.eval()

    print()
    print("=" * 70)
    print(
        f"SAMPLES @ STEP "
        f"{step:,}"
    )
    print("=" * 70)

    for prompt in SAMPLE_PROMPTS:

        encoding = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        token_ids = torch.tensor(
            [
                encoding.ids
            ],
            dtype=torch.long,
            device=device,
        )

        generated = model.generate(
            token_ids,
            max_new_tokens=(
                SAMPLE_NEW_TOKENS
            ),
            temperature=0.8,
            top_k=50,
        )

        output = tokenizer.decode(
            generated[
                0
            ].tolist(),
            skip_special_tokens=False,
        )

        # For readability stop display after EOT.
        if "<|END_OF_TEXT|>" in output:

            output = (
                output.split(
                    "<|END_OF_TEXT|>",
                    1,
                )[0]
            )

        print()
        print(
            f"Prompt: "
            f"{prompt}"
        )

        print(
            f"Output: "
            f"{output}"
        )

    print()

    model.train()


# ============================================================
# Checkpoints
# ============================================================

def save_latest_checkpoint(
    model,
    optimizer,
    global_step,
    epoch,
    position_in_epoch,
    tokens_seen,
    best_val_loss,
    max_steps,
):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        CHECKPOINT_DIR
        / "latest.pt.tmp"
    )

    payload = {
        "version":
            3,

        "model":
            model.state_dict(),

        "optimizer":
            optimizer.state_dict(),

        "global_step":
            global_step,

        "epoch":
            epoch,

        "position_in_epoch":
            position_in_epoch,

        "tokens_seen":
            tokens_seen,

        "best_val_loss":
            best_val_loss,

        "max_steps":
            max_steps,

        "config": {
            "vocab_size":
                VOCAB_SIZE,

            "block_size":
                BLOCK_SIZE,

            "batch_size":
                BATCH_SIZE,
        },
    }

    torch.save(
        payload,
        temporary_path,
    )

    os.replace(
        temporary_path,
        LATEST_CHECKPOINT,
    )


def save_model_only(
    path,
    model,
    global_step,
    tokens_seen,
    val_loss,
):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = Path(
        str(path)
        + ".tmp"
    )

    payload = {
        "version":
            3,

        "model":
            model.state_dict(),

        "global_step":
            global_step,

        "tokens_seen":
            tokens_seen,

        "val_loss":
            val_loss,
    }

    torch.save(
        payload,
        temporary_path,
    )

    os.replace(
        temporary_path,
        path,
    )


def load_checkpoint(
    path,
):

    print()
    print(
        f"Loading checkpoint: "
        f"{path}"
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get(
        "version"
    ) != 3:

        raise RuntimeError(
            "Unsupported checkpoint version."
        )

    return checkpoint


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a short training "
            "smoke test."
        ),
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Ignore latest checkpoint "
            "and start from random weights."
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    set_seed(
        SEED
    )

    if not torch.backends.mps.is_available():

        raise RuntimeError(
            "Apple MPS is required "
            "for this training run."
        )

    device = torch.device(
        "mps"
    )

    torch.mps.empty_cache()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_dataset = (
        BinaryTokenDataset(
            TRAIN_DATA_PATH
        )
    )

    validation_dataset = (
        BinaryTokenDataset(
            VALIDATION_DATA_PATH
        )
    )

    steps_per_epoch = (
        math.ceil(
            train_dataset.block_count
            / BATCH_SIZE
        )
    )

    default_max_steps = (
        steps_per_epoch
    )

    if args.max_steps is None:

        max_steps = (
            default_max_steps
        )

    else:

        max_steps = (
            args.max_steps
        )

    if max_steps <= 0:

        raise ValueError(
            "max_steps must be > 0"
        )

    # --------------------------------------------------------
    # Smoke mode
    # --------------------------------------------------------

    if args.smoke:

        max_steps = 20

        warmup_steps = 5

        eval_interval = 10

        eval_batches = 3

        checkpoint_interval = None

        sample_interval = 20

        log_interval = 1

    else:

        warmup_steps = min(
            WARMUP_STEPS,
            max_steps,
        )

        eval_interval = (
            EVAL_INTERVAL
        )

        eval_batches = (
            EVAL_BATCHES
        )

        checkpoint_interval = (
            CHECKPOINT_INTERVAL
        )

        sample_interval = (
            SAMPLE_INTERVAL
        )

        log_interval = (
            LOG_INTERVAL
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("=" * 72)
    print(
        "MiniGPT-M4 V3 — 80M Pretraining"
    )
    print("=" * 72)

    print()
    print(
        f"Device              : "
        f"{device}"
    )

    print(
        f"Vocabulary          : "
        f"{VOCAB_SIZE:,}"
    )

    print(
        f"Context             : "
        f"{BLOCK_SIZE}"
    )

    print(
        f"Batch size          : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Tokens / full batch : "
        f"{BATCH_SIZE * BLOCK_SIZE:,}"
    )

    print()
    print(
        f"Train tokens        : "
        f"{train_dataset.token_count:,}"
    )

    print(
        f"Train blocks        : "
        f"{train_dataset.block_count:,}"
    )

    print(
        f"Validation tokens   : "
        f"{validation_dataset.token_count:,}"
    )

    print(
        f"Steps / epoch       : "
        f"{steps_per_epoch:,}"
    )

    print(
        f"Maximum steps       : "
        f"{max_steps:,}"
    )

    print(
        f"Warmup steps        : "
        f"{warmup_steps:,}"
    )

    if args.smoke:

        print()
        print(
            "MODE                : "
            "SMOKE TEST"
        )

    # --------------------------------------------------------
    # Resume state
    # --------------------------------------------------------

    checkpoint = None

    if (
        not args.smoke
        and
        not args.fresh
        and
        LATEST_CHECKPOINT.exists()
    ):

        checkpoint = (
            load_checkpoint(
                LATEST_CHECKPOINT
            )
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MiniGPTV3()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        f"Parameters          : "
        f"{parameter_count:,}"
    )

    print(
        f"Parameters M        : "
        f"{parameter_count / 1e6:.4f}M"
    )

    if checkpoint is not None:

        model.load_state_dict(
            checkpoint[
                "model"
            ]
        )

    model = model.to(
        device
    )

    model.train()

    optimizer = build_optimizer(
        model
    )

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    global_step = 0
    epoch = 0
    position_in_epoch = 0
    tokens_seen = 0

    best_val_loss = float(
        "inf"
    )

    if checkpoint is not None:

        optimizer.load_state_dict(
            checkpoint[
                "optimizer"
            ]
        )

        move_optimizer_to_device(
            optimizer,
            device,
        )

        global_step = int(
            checkpoint[
                "global_step"
            ]
        )

        epoch = int(
            checkpoint[
                "epoch"
            ]
        )

        position_in_epoch = int(
            checkpoint[
                "position_in_epoch"
            ]
        )

        tokens_seen = int(
            checkpoint[
                "tokens_seen"
            ]
        )

        best_val_loss = float(
            checkpoint.get(
                "best_val_loss",
                float("inf"),
            )
        )

        print()
        print("=" * 72)
        print("RESUME")
        print("=" * 72)

        print()
        print(
            f"Global step         : "
            f"{global_step:,}"
        )

        print(
            f"Epoch               : "
            f"{epoch}"
        )

        print(
            f"Position in epoch   : "
            f"{position_in_epoch:,}"
        )

        print(
            f"Tokens seen         : "
            f"{tokens_seen:,}"
        )

        print(
            f"Best val loss       : "
            f"{best_val_loss:.4f}"
        )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = Tokenizer.from_file(
        str(
            TOKENIZER_PATH
        )
    )

    # --------------------------------------------------------
    # Initial validation
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("INITIAL VALIDATION")
    print("=" * 72)

    initial_val_loss = evaluate(
        model,
        validation_dataset,
        device,
        BATCH_SIZE,
        eval_batches,
    )

    print()
    print(
        f"Validation loss     : "
        f"{initial_val_loss:.4f}"
    )

    print(
        f"Perplexity          : "
        f"{math.exp(min(initial_val_loss, 20)):.2f}"
    )

    if (
        best_val_loss
        == float("inf")
    ):

        best_val_loss = (
            initial_val_loss
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("TRAINING")
    print("=" * 72)
    print()

    training_start = (
        time.perf_counter()
    )

    interval_start = (
        time.perf_counter()
    )

    interval_tokens = 0

    permutation = None

    try:

        while global_step < max_steps:

            # ------------------------------------------------
            # New epoch permutation
            # ------------------------------------------------

            if (
                permutation is None
            ):

                rng = (
                    np.random.default_rng(
                        SEED + epoch
                    )
                )

                permutation = (
                    rng.permutation(
                        train_dataset
                        .block_count
                    )
                )

            # ------------------------------------------------
            # Epoch finished
            # ------------------------------------------------

            if (
                position_in_epoch
                >= train_dataset.block_count
            ):

                epoch += 1

                position_in_epoch = 0

                permutation = None

                continue

            # ------------------------------------------------
            # Batch IDs
            # ------------------------------------------------

            batch_end = min(
                position_in_epoch
                + BATCH_SIZE,
                train_dataset.block_count,
            )

            block_ids = permutation[
                position_in_epoch:
                batch_end
            ]

            actual_batch_size = len(
                block_ids
            )

            x, y = (
                train_dataset.get_batch(
                    block_ids,
                    device,
                )
            )

            # ------------------------------------------------
            # LR
            # ------------------------------------------------

            learning_rate = (
                get_learning_rate(
                    global_step,
                    max_steps,
                    warmup_steps,
                )
            )

            for group in (
                optimizer.param_groups
            ):

                group[
                    "lr"
                ] = learning_rate

            # ------------------------------------------------
            # Train step
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            _, loss = model(
                x,
                y,
            )

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    f"Non-finite loss "
                    f"at step "
                    f"{global_step:,}: "
                    f"{loss.item()}"
                )

            loss.backward()

            grad_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )
            )

            if not torch.isfinite(
                grad_norm
            ):

                raise RuntimeError(
                    f"Non-finite gradient "
                    f"at step "
                    f"{global_step:,}"
                )

            optimizer.step()

            # ------------------------------------------------
            # Counters
            # ------------------------------------------------

            batch_tokens = (
                actual_batch_size
                * BLOCK_SIZE
            )

            tokens_seen += (
                batch_tokens
            )

            interval_tokens += (
                batch_tokens
            )

            position_in_epoch = (
                batch_end
            )

            global_step += 1

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            if (
                global_step
                % log_interval
                == 0
                or global_step == 1
                or global_step == max_steps
            ):

                if hasattr(
                    torch.mps,
                    "synchronize",
                ):

                    torch.mps.synchronize()

                now = (
                    time.perf_counter()
                )

                interval_elapsed = (
                    now
                    - interval_start
                )

                throughput = (
                    interval_tokens
                    / max(
                        interval_elapsed,
                        1e-9,
                    )
                )

                elapsed_total = (
                    now
                    - training_start
                )

                average_step_time = (
                    elapsed_total
                    / max(
                        1,
                        global_step
                        - (
                            checkpoint[
                                "global_step"
                            ]
                            if checkpoint
                            else 0
                        ),
                    )
                )

                remaining_steps = (
                    max_steps
                    - global_step
                )

                eta = (
                    average_step_time
                    * remaining_steps
                )

                allocated, driver = (
                    mps_memory()
                )

                memory_text = ""

                if allocated is not None:

                    memory_text += (
                        f" | mps "
                        f"{allocated:.2f}G"
                    )

                if driver is not None:

                    memory_text += (
                        f"/{driver:.2f}G"
                    )

                print(
                    f"step "
                    f"{global_step:>7,}"
                    f"/"
                    f"{max_steps:,}"
                    f" | loss "
                    f"{loss.item():.4f}"
                    f" | grad "
                    f"{float(grad_norm):.3f}"
                    f" | lr "
                    f"{learning_rate:.2e}"
                    f" | "
                    f"{throughput:,.0f} tok/s"
                    f" | seen "
                    f"{tokens_seen / 1e6:.2f}M"
                    f" | ETA "
                    f"{format_duration(eta)}"
                    f"{memory_text}"
                )

                interval_start = now

                interval_tokens = 0

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            if (
                global_step
                % eval_interval
                == 0
                or global_step == max_steps
            ):

                val_start = (
                    time.perf_counter()
                )

                val_loss = evaluate(
                    model,
                    validation_dataset,
                    device,
                    BATCH_SIZE,
                    eval_batches,
                )

                val_elapsed = (
                    time.perf_counter()
                    - val_start
                )

                perplexity = (
                    math.exp(
                        min(
                            val_loss,
                            20,
                        )
                    )
                )

                print()
                print(
                    f"[VAL] "
                    f"step="
                    f"{global_step:,}"
                    f" "
                    f"loss="
                    f"{val_loss:.4f}"
                    f" "
                    f"ppl="
                    f"{perplexity:.2f}"
                    f" "
                    f"time="
                    f"{val_elapsed:.1f}s"
                )

                if (
                    val_loss
                    < best_val_loss
                ):

                    best_val_loss = (
                        val_loss
                    )

                    if not args.smoke:

                        save_model_only(
                            BEST_MODEL,
                            model,
                            global_step,
                            tokens_seen,
                            val_loss,
                        )

                        print(
                            f"[BEST] "
                            f"saved "
                            f"{BEST_MODEL}"
                        )

                print()

            # ------------------------------------------------
            # Sample
            # ------------------------------------------------

            if (
                global_step
                % sample_interval
                == 0
                or global_step == max_steps
            ):

                generate_samples(
                    model,
                    tokenizer,
                    device,
                    global_step,
                )

            # ------------------------------------------------
            # Resume checkpoint
            # ------------------------------------------------

            if (
                not args.smoke
                and
                checkpoint_interval
                is not None
                and
                (
                    global_step
                    % checkpoint_interval
                    == 0
                    or
                    global_step
                    == max_steps
                )
            ):

                checkpoint_start = (
                    time.perf_counter()
                )

                save_latest_checkpoint(
                    model,
                    optimizer,
                    global_step,
                    epoch,
                    position_in_epoch,
                    tokens_seen,
                    best_val_loss,
                    max_steps,
                )

                checkpoint_elapsed = (
                    time.perf_counter()
                    - checkpoint_start
                )

                print(
                    f"[CKPT] "
                    f"saved "
                    f"{LATEST_CHECKPOINT}"
                    f" "
                    f"({checkpoint_elapsed:.1f}s)"
                )

    except KeyboardInterrupt:

        print()
        print()
        print("=" * 72)
        print(
            "TRAINING INTERRUPTED"
        )
        print("=" * 72)

        if not args.smoke:

            print()
            print(
                "Saving resume checkpoint..."
            )

            save_latest_checkpoint(
                model,
                optimizer,
                global_step,
                epoch,
                position_in_epoch,
                tokens_seen,
                best_val_loss,
                max_steps,
            )

            print(
                f"Saved: "
                f"{LATEST_CHECKPOINT}"
            )

        print()
        print(
            "Run train_v3.py again "
            "to resume."
        )

        return

    # --------------------------------------------------------
    # Final model
    # --------------------------------------------------------

    total_elapsed = (
        time.perf_counter()
        - training_start
    )

    if not args.smoke:

        final_val_loss = evaluate(
            model,
            validation_dataset,
            device,
            BATCH_SIZE,
            eval_batches,
        )

        save_model_only(
            FINAL_MODEL,
            model,
            global_step,
            tokens_seen,
            final_val_loss,
        )

        save_latest_checkpoint(
            model,
            optimizer,
            global_step,
            epoch,
            position_in_epoch,
            tokens_seen,
            best_val_loss,
            max_steps,
        )

    print()
    print("=" * 72)

    if args.smoke:

        print(
            "SMOKE TEST PASSED ✓"
        )

    else:

        print(
            "PRETRAINING COMPLETE ✓"
        )

    print("=" * 72)

    print()
    print(
        f"Final step          : "
        f"{global_step:,}"
    )

    print(
        f"Tokens seen         : "
        f"{tokens_seen:,}"
    )

    print(
        f"Best validation     : "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Training time       : "
        f"{format_duration(total_elapsed)}"
    )

    if not args.smoke:

        print()
        print(
            f"Latest checkpoint   : "
            f"{LATEST_CHECKPOINT}"
        )

        print(
            f"Best model          : "
            f"{BEST_MODEL}"
        )

        print(
            f"Final model         : "
            f"{FINAL_MODEL}"
        )


if __name__ == "__main__":
    main()
