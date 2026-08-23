import math
import time
from pathlib import Path

import torch

from config import (
    BATCH_SIZE,
    BLOCK_SIZE,
    CHECKPOINT_INTERVAL,
    EVAL_INTERVAL,
    EVAL_ITERS,
    GRAD_CLIP,
    LEARNING_RATE,
    MAX_STEPS,
    MIN_LEARNING_RATE,
    SAMPLE_INTERVAL,
    SAMPLE_TOKENS,
    SEED,
    VOCAB_SIZE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)

from dataset import TokenDataset
from model.gpt import MiniGPT
from tokenizer.bpe import BPETokenizer


CHECKPOINT_DIR = Path("checkpoints")

TOKENIZER_PATH = Path(
    "tokenizer/tokenizer.json"
)


def get_device() -> torch.device:

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def synchronize(
    device: torch.device,
) -> None:

    if (
        device.type == "mps"
        and hasattr(torch, "mps")
    ):
        torch.mps.synchronize()


def count_parameters(
    model: torch.nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )


def get_learning_rate(
    step: int,
) -> float:

    # ----------------------------------------------
    # Linear warmup
    # ----------------------------------------------

    if step < WARMUP_STEPS:

        return (
            LEARNING_RATE
            * (step + 1)
            / WARMUP_STEPS
        )

    # ----------------------------------------------
    # Cosine decay
    # ----------------------------------------------

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


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    dataset: TokenDataset,
    device: torch.device,
) -> dict[str, float]:

    model.eval()

    losses = {}

    for split in (
        "train",
        "validation",
    ):

        split_losses = []

        for _ in range(
            EVAL_ITERS
        ):

            x, y = dataset.get_batch(
                split=split,
                device=device,
            )

            _, loss = model(
                x,
                y,
            )

            split_losses.append(
                loss.item()
            )

        losses[split] = (
            sum(split_losses)
            / len(split_losses)
        )

    model.train()

    return losses


@torch.no_grad()
def generate_sample(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    device: torch.device,
) -> str:

    model.eval()

    prompt = (
        "Once upon a time, "
        "there was a little"
    )

    prompt_tokens = (
        tokenizer.encode(prompt)
    )

    token_tensor = torch.tensor(
        [prompt_tokens],
        dtype=torch.long,
        device=device,
    )

    generated = model.generate(
        token_tensor,
        max_new_tokens=SAMPLE_TOKENS,
        temperature=0.8,
        top_k=50,
    )

    generated_text = (
        tokenizer.decode(
            generated[0]
            .cpu()
            .tolist()
        )
    )

    model.train()

    return generated_text


def save_checkpoint(
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    validation_loss: float | None,
) -> Path:

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        CHECKPOINT_DIR
        / f"step_{step:06d}.pt"
    )

    checkpoint = {
        "step": step,
        "model_state_dict":
            model.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict(),
        "validation_loss":
            validation_loss,
    }

    torch.save(
        checkpoint,
        path,
    )

    return path


def main():

    torch.manual_seed(
        SEED
    )

    device = get_device()

    print("=" * 60)
    print("MiniGPT-M4 Training")
    print("=" * 60)

    print()
    print(
        f"Device       : {device}"
    )

    print(
        f"Batch size   : {BATCH_SIZE}"
    )

    print(
        f"Block size   : {BLOCK_SIZE}"
    )

    print(
        f"Tokens/step  : "
        f"{BATCH_SIZE * BLOCK_SIZE:,}"
    )

    print(
        f"Max steps    : "
        f"{MAX_STEPS:,}"
    )

    print(
        f"Vocabulary   : "
        f"{VOCAB_SIZE:,}"
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    print()

    dataset = TokenDataset()

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    tokenizer = BPETokenizer.load(
        TOKENIZER_PATH
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = MiniGPT().to(
        device
    )

    parameter_count = (
        count_parameters(model)
    )

    print()
    print(
        f"Parameters   : "
        f"{parameter_count:,}"
    )

    print(
        f"Model size   : "
        f"{parameter_count / 1_000_000:.3f}M"
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )

    CHECKPOINT_DIR.mkdir(
        exist_ok=True
    )

    print()
    print("=" * 60)
    print("INITIAL EVALUATION")
    print("=" * 60)

    losses = estimate_loss(
        model,
        dataset,
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

    print()
    print("=" * 60)
    print("TRAINING START")
    print("=" * 60)

    model.train()

    running_loss = 0.0
    running_steps = 0

    last_validation_loss = None

    total_tokens_seen = 0

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

        # ------------------------------------------
        # Learning rate
        # ------------------------------------------

        learning_rate = (
            get_learning_rate(
                step - 1
            )
        )

        for parameter_group in (
            optimizer.param_groups
        ):

            parameter_group["lr"] = (
                learning_rate
            )

        # ------------------------------------------
        # Batch
        # ------------------------------------------

        x, y = dataset.get_batch(
            split="train",
            device=device,
        )

        # ------------------------------------------
        # Forward + backward
        # ------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        synchronize(device)

        start_time = time.perf_counter()

        _, loss = model(
            x,
            y,
        )

        loss.backward()

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )
        )

        optimizer.step()

        synchronize(device)

        elapsed = (
            time.perf_counter()
            - start_time
        )

        # ------------------------------------------
        # Statistics
        # ------------------------------------------

        tokens_this_step = (
            BATCH_SIZE
            * BLOCK_SIZE
        )

        total_tokens_seen += (
            tokens_this_step
        )

        tokens_per_second = (
            tokens_this_step
            / elapsed
        )

        running_loss += (
            loss.item()
        )

        running_steps += 1

        # ------------------------------------------
        # Console log
        # ------------------------------------------

        if (
            step == 1
            or step % 25 == 0
        ):

            average_loss = (
                running_loss
                / running_steps
            )

            print(
                f"step {step:>5}/{MAX_STEPS}"
                f" | loss {average_loss:.4f}"
                f" | lr {learning_rate:.2e}"
                f" | grad {float(gradient_norm):.3f}"
                f" | tok/s {tokens_per_second:,.0f}"
                f" | seen {total_tokens_seen / 1_000_000:.2f}M"
            )

            running_loss = 0.0
            running_steps = 0

        # ------------------------------------------
        # Validation
        # ------------------------------------------

        if (
            step % EVAL_INTERVAL == 0
            or step == MAX_STEPS
        ):

            print()
            print("-" * 60)

            losses = estimate_loss(
                model,
                dataset,
                device,
            )

            last_validation_loss = (
                losses["validation"]
            )

            print(
                f"Evaluation @ step {step}"
            )

            print(
                f"Train loss      : "
                f"{losses['train']:.4f}"
            )

            print(
                f"Validation loss : "
                f"{losses['validation']:.4f}"
            )

            print("-" * 60)
            print()

        # ------------------------------------------
        # Sample generation
        # ------------------------------------------

        if (
            step % SAMPLE_INTERVAL == 0
            or step == MAX_STEPS
        ):

            sample = generate_sample(
                model,
                tokenizer,
                device,
            )

            print()
            print("=" * 60)
            print(
                f"SAMPLE @ STEP {step}"
            )
            print("=" * 60)

            print(sample)

            print()
            print("=" * 60)
            print()

        # ------------------------------------------
        # Checkpoint
        # ------------------------------------------

        if (
            step % CHECKPOINT_INTERVAL == 0
            or step == MAX_STEPS
        ):

            checkpoint_path = (
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    last_validation_loss,
                )
            )

            print(
                f"Checkpoint saved: "
                f"{checkpoint_path}"
            )

    # --------------------------------------------------
    # Final
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING COMPLETE ✓")
    print("=" * 60)

    print(
        f"Total tokens seen: "
        f"{total_tokens_seen:,}"
    )

    print(
        f"Final validation loss: "
        f"{last_validation_loss}"
    )


if __name__ == "__main__":
    main()