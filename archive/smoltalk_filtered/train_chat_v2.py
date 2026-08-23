import math
import time
from pathlib import Path

import torch

from model.gpt import MiniGPT
from smoltalk_dataset import SmolTalkDataset
from tokenizer.bpe import BPETokenizer


# ============================================================
# Configuration
# ============================================================

BASE_CHECKPOINT = Path(
    "checkpoints/step_005000.pt"
)

CHECKPOINT_DIR = Path(
    "chat_checkpoints_v2"
)

TOKENIZER_PATH = Path(
    "tokenizer/tokenizer.json"
)

MAX_STEPS = 2000

LEARNING_RATE = 3e-5
MIN_LEARNING_RATE = 3e-6

WARMUP_STEPS = 100

WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

EVAL_INTERVAL = 100
EVAL_ITERS = 40

CHECKPOINT_INTERVAL = 250
SAMPLE_INTERVAL = 250

EARLY_STOP_PATIENCE = 6

SEED = 42

TEMPERATURE = 0.7
TOP_K = 40
SAMPLE_TOKENS = 80


# ============================================================
# Fixed evaluation prompts
# ============================================================

TEST_PROMPTS = [
    "Hello! How are you?",
    "My name is Fatih. What is your name?",
    "I like computers. What do you like?",
]


# ============================================================
# Device
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


# ============================================================
# LR schedule
# ============================================================

def get_learning_rate(step):

    if step < WARMUP_STEPS:
        return (
            LEARNING_RATE
            * (step + 1)
            / WARMUP_STEPS
        )

    if step >= MAX_STEPS:
        return MIN_LEARNING_RATE

    decay_ratio = (
        (step - WARMUP_STEPS)
        /
        (MAX_STEPS - WARMUP_STEPS)
    )

    coefficient = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * decay_ratio
            )
        )
    )

    return (
        MIN_LEARNING_RATE
        + coefficient
        * (
            LEARNING_RATE
            - MIN_LEARNING_RATE
        )
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def estimate_loss(
    model,
    train_dataset,
    validation_dataset,
    device,
):

    model.eval()

    results = {}

    for name, dataset in (
        ("train", train_dataset),
        ("validation", validation_dataset),
    ):

        losses = []

        for _ in range(EVAL_ITERS):

            x, y = dataset.get_batch(
                device=device
            )

            _, loss = model(
                x,
                y,
            )

            losses.append(
                loss.item()
            )

        results[name] = (
            sum(losses)
            / len(losses)
        )

    model.train()

    return results


# ============================================================
# Generation helper
# ============================================================

@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    user_text,
    device,
):

    prompt = (
        "<|USER|>\n"
        + user_text.strip()
        + "\n<|END_OF_TURN|>\n"
        + "<|ASSISTANT|>\n"
    )

    token_ids = tokenizer.encode(
        prompt
    )

    x = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    model.eval()

    generated = model.generate(
        x,
        max_new_tokens=SAMPLE_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )

    generated_tokens = generated[
        0,
        len(token_ids):
    ]

    text = tokenizer.decode(
        generated_tokens
        .cpu()
        .tolist()
    )

    model.train()

    # Stop at EOT
    if "<|END_OF_TURN|>" in text:
        text = text.split(
            "<|END_OF_TURN|>",
            1,
        )[0]

    if "<|USER|>" in text:
        text = text.split(
            "<|USER|>",
            1,
        )[0]

    if "<|ASSISTANT|>" in text:
        text = text.split(
            "<|ASSISTANT|>",
            1,
        )[0]

    return text.strip()


# ============================================================
# Sample suite
# ============================================================

def print_samples(
    model,
    tokenizer,
    device,
    step,
):

    print()
    print("=" * 60)
    print(
        f"CHAT SAMPLES @ STEP {step}"
    )
    print("=" * 60)

    for prompt in TEST_PROMPTS:

        response = generate_response(
            model,
            tokenizer,
            prompt,
            device,
        )

        print()
        print(
            f"USER: {prompt}"
        )

        print(
            f"MINIGPT: {response}"
        )

    print()
    print("=" * 60)


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    step,
    validation_loss,
):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "step": step,
            "base_checkpoint":
                str(BASE_CHECKPOINT),

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "validation_loss":
                validation_loss,
        },
        path,
    )


# ============================================================
# Main
# ============================================================

def main():

    torch.manual_seed(
        SEED
    )

    device = get_device()

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("MiniGPT-M4 Chat V2 Fine-Tuning")
    print("=" * 60)

    print()
    print(
        f"Device          : {device}"
    )

    print(
        f"Base checkpoint : "
        f"{BASE_CHECKPOINT}"
    )

    print(
        f"Max steps       : "
        f"{MAX_STEPS:,}"
    )

    print(
        f"Peak LR         : "
        f"{LEARNING_RATE:.2e}"
    )

    # ========================================================
    # Dataset
    # ========================================================

    print()

    train_dataset = (
        SmolTalkDataset(
            "train"
        )
    )

    print()

    validation_dataset = (
        SmolTalkDataset(
            "validation"
        )
    )

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer = BPETokenizer.load(
        TOKENIZER_PATH
    )

    # ========================================================
    # Model
    # ========================================================

    print()
    print(
        "Loading clean story checkpoint..."
    )

    checkpoint = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model = MiniGPT()

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        device
    )

    print(
        f"Loaded base step: "
        f"{checkpoint['step']}"
    )

    print(
        f"Base validation loss: "
        f"{checkpoint['validation_loss']}"
    )

    # IMPORTANT:
    #
    # Fresh optimizer.
    #
    # DailyDialog optimizer state'i yok.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(
            0.9,
            0.95,
        ),
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # Initial validation
    # ========================================================

    print()
    print("=" * 60)
    print("INITIAL CHAT V2 EVALUATION")
    print("=" * 60)

    losses = estimate_loss(
        model,
        train_dataset,
        validation_dataset,
        device,
    )

    print(
        f"Train loss      : "
        f"{losses['train']:.4f}"
    )

    print(
        f"Validation loss : "
        f"{losses['validation']:.4f}"
    )

    print_samples(
        model,
        tokenizer,
        device,
        step=0,
    )

    # ========================================================
    # Best model state
    # ========================================================

    best_validation_loss = float(
        "inf"
    )

    best_step = 0

    evals_without_improvement = 0

    # ========================================================
    # Training
    # ========================================================

    model.train()

    running_loss = 0.0
    running_steps = 0

    print()
    print("=" * 60)
    print("CHAT V2 TRAINING START")
    print("=" * 60)

    stop_training = False

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

        # ----------------------------------------------------
        # Learning rate
        # ----------------------------------------------------

        lr = get_learning_rate(
            step - 1
        )

        for group in (
            optimizer.param_groups
        ):
            group["lr"] = lr

        # ----------------------------------------------------
        # Batch
        # ----------------------------------------------------

        x, y = train_dataset.get_batch(
            device=device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        synchronize(
            device
        )

        start_time = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        _, loss = model(
            x,
            y,
        )

        # ----------------------------------------------------
        # Backprop
        # ----------------------------------------------------

        loss.backward()

        gradient_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )
        )

        optimizer.step()

        synchronize(
            device
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        running_loss += (
            loss.item()
        )

        running_steps += 1

        if (
            step == 1
            or step % 25 == 0
        ):

            average_loss = (
                running_loss
                / running_steps
            )

            supervised_tokens = (
                y != -100
            ).sum().item()

            target_tokens_per_second = (
                supervised_tokens
                / elapsed
            )

            print(
                f"step "
                f"{step:>4}/{MAX_STEPS}"
                f" | loss "
                f"{average_loss:.4f}"
                f" | lr "
                f"{lr:.2e}"
                f" | grad "
                f"{float(gradient_norm):.3f}"
                f" | target tok/s "
                f"{target_tokens_per_second:,.0f}"
            )

            running_loss = 0.0
            running_steps = 0

        # ====================================================
        # Evaluation
        # ====================================================

        if (
            step % EVAL_INTERVAL == 0
            or step == MAX_STEPS
        ):

            losses = estimate_loss(
                model,
                train_dataset,
                validation_dataset,
                device,
            )

            validation_loss = (
                losses[
                    "validation"
                ]
            )

            print()
            print("-" * 60)

            print(
                f"Evaluation @ "
                f"step {step}"
            )

            print(
                f"Train loss      : "
                f"{losses['train']:.4f}"
            )

            print(
                f"Validation loss : "
                f"{validation_loss:.4f}"
            )

            print("-" * 60)

            # ------------------------------------------------
            # BEST CHECKPOINT
            # ------------------------------------------------

            if (
                validation_loss
                < best_validation_loss
            ):

                best_validation_loss = (
                    validation_loss
                )

                best_step = step

                evals_without_improvement = 0

                best_path = (
                    CHECKPOINT_DIR
                    / "best.pt"
                )

                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    step,
                    validation_loss,
                )

                print()
                print(
                    "New best checkpoint ✓"
                )

                print(
                    f"Step     : "
                    f"{best_step}"
                )

                print(
                    f"Val loss : "
                    f"{best_validation_loss:.4f}"
                )

            else:

                evals_without_improvement += 1

                print()
                print(
                    "No validation improvement."
                )

                print(
                    f"Patience: "
                    f"{evals_without_improvement}"
                    f"/{EARLY_STOP_PATIENCE}"
                )

            # ------------------------------------------------
            # Early stopping
            # ------------------------------------------------

            if (
                evals_without_improvement
                >= EARLY_STOP_PATIENCE
            ):

                print()
                print(
                    "Early stopping triggered."
                )

                stop_training = True

        # ====================================================
        # Samples
        # ====================================================

        if (
            step % SAMPLE_INTERVAL == 0
            or step == MAX_STEPS
        ):

            print_samples(
                model,
                tokenizer,
                device,
                step,
            )

        # ====================================================
        # Periodic checkpoint
        # ====================================================

        if (
            step
            % CHECKPOINT_INTERVAL
            == 0
        ):

            checkpoint_path = (
                CHECKPOINT_DIR
                / (
                    f"step_"
                    f"{step:06d}.pt"
                )
            )

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                step,
                losses.get(
                    "validation",
                    None,
                ),
            )

            print()
            print(
                f"Checkpoint saved: "
                f"{checkpoint_path}"
            )

        if stop_training:
            break

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 60)
    print("CHAT V2 TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best step            : "
        f"{best_step}"
    )

    print(
        f"Best validation loss : "
        f"{best_validation_loss:.4f}"
    )

    print(
        f"Best checkpoint      : "
        f"{CHECKPOINT_DIR / 'best.pt'}"
    )


if __name__ == "__main__":
    main()