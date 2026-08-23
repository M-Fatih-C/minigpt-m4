import math
import time
from pathlib import Path

import torch

from chat_dataset import ChatDataset
from model.gpt import MiniGPT
from tokenizer.bpe import BPETokenizer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_CHECKPOINT = Path(
    "checkpoints/step_005000.pt"
)

CHECKPOINT_DIR = Path(
    "chat_checkpoints"
)

TOKENIZER_PATH = Path(
    "tokenizer/tokenizer.json"
)

MAX_STEPS = 1500

LEARNING_RATE = 5e-5
MIN_LEARNING_RATE = 5e-6

WARMUP_STEPS = 100

WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

EVAL_INTERVAL = 100
EVAL_ITERS = 30

CHECKPOINT_INTERVAL = 250
SAMPLE_INTERVAL = 250

SEED = 42


# --------------------------------------------------
# Device
# --------------------------------------------------

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def synchronize(device):

    if device.type == "mps":
        torch.mps.synchronize()


# --------------------------------------------------
# Learning rate
# --------------------------------------------------

def get_learning_rate(step):

    if step < WARMUP_STEPS:

        return (
            LEARNING_RATE
            * (step + 1)
            / WARMUP_STEPS
        )

    if step >= MAX_STEPS:
        return MIN_LEARNING_RATE

    ratio = (
        (step - WARMUP_STEPS)
        /
        (MAX_STEPS - WARMUP_STEPS)
    )

    coefficient = (
        0.5
        * (
            1
            + math.cos(
                math.pi * ratio
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


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

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


# --------------------------------------------------
# Chat sample
# --------------------------------------------------

@torch.no_grad()
def generate_sample(
    model,
    tokenizer,
    device,
):

    prompt = (
        "<|USER|>\n"
        "Hi! How are you today?"
        "\n<|END_OF_TURN|>\n"
        "<|ASSISTANT|>\n"
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
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
    )

    text = tokenizer.decode(
        generated[0]
        .cpu()
        .tolist()
    )

    model.train()

    # Sadece assistant cevabını göstermek için
    # END_OF_TURN sonrasında kesebiliriz.
    assistant_start = (
        "<|ASSISTANT|>\n"
    )

    if assistant_start in text:

        answer = text.split(
            assistant_start,
            1,
        )[1]

    else:
        answer = text

    if "<|END_OF_TURN|>" in answer:

        answer = answer.split(
            "<|END_OF_TURN|>",
            1,
        )[0]

    return answer.strip()


# --------------------------------------------------
# Checkpoint
# --------------------------------------------------

def save_checkpoint(
    model,
    optimizer,
    step,
    validation_loss,
):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        CHECKPOINT_DIR
        / f"chat_step_{step:06d}.pt"
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

    return path


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    torch.manual_seed(SEED)

    device = get_device()

    print("=" * 60)
    print("MiniGPT-M4 Chat Fine-Tuning")
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

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    print()

    train_dataset = ChatDataset(
        "train"
    )

    validation_dataset = ChatDataset(
        "validation"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print()
    print(
        "Loading pretrained story model..."
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

    model.to(device)

    print(
        f"Loaded base step: "
        f"{checkpoint['step']}"
    )

    print(
        f"Base validation loss: "
        f"{checkpoint['validation_loss']}"
    )

    # --------------------------------------------------
    # IMPORTANT:
    # Fresh optimizer for SFT.
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------
    # Initial chat loss
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("INITIAL CHAT EVALUATION")
    print("=" * 60)

    losses = estimate_loss(
        model,
        train_dataset,
        validation_dataset,
        device,
    )

    print(
        f"Train chat loss      : "
        f"{losses['train']:.4f}"
    )

    print(
        f"Validation chat loss : "
        f"{losses['validation']:.4f}"
    )

    print()
    print(
        "Initial sample:"
    )

    print(
        generate_sample(
            model,
            tokenizer=BPETokenizer.load(
                TOKENIZER_PATH
            ),
            device=device,
        )
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    tokenizer = BPETokenizer.load(
        TOKENIZER_PATH
    )

    model.train()

    running_loss = 0.0
    running_steps = 0

    last_validation_loss = None

    print()
    print("=" * 60)
    print("CHAT FINE-TUNING START")
    print("=" * 60)

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

        lr = get_learning_rate(
            step - 1
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = train_dataset.get_batch(
            device=device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        synchronize(device)

        start = time.perf_counter()

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
            - start
        )

        running_loss += loss.item()
        running_steps += 1

        if (
            step == 1
            or step % 25 == 0
        ):

            average_loss = (
                running_loss
                / running_steps
            )

            # Gerçek supervised target sayısı.
            supervised_tokens = (
                y != -100
            ).sum().item()

            target_tokens_per_second = (
                supervised_tokens
                / elapsed
            )

            print(
                f"step {step:>4}/{MAX_STEPS}"
                f" | loss {average_loss:.4f}"
                f" | lr {lr:.2e}"
                f" | grad {float(gradient_norm):.3f}"
                f" | target tok/s "
                f"{target_tokens_per_second:,.0f}"
            )

            running_loss = 0.0
            running_steps = 0

        # ------------------------------------------
        # Evaluation
        # ------------------------------------------

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

            last_validation_loss = (
                losses["validation"]
            )

            print()
            print("-" * 60)

            print(
                f"Evaluation @ "
                f"chat step {step}"
            )

            print(
                f"Train chat loss      : "
                f"{losses['train']:.4f}"
            )

            print(
                f"Validation chat loss : "
                f"{losses['validation']:.4f}"
            )

            print("-" * 60)
            print()

        # ------------------------------------------
        # Sample
        # ------------------------------------------

        if (
            step % SAMPLE_INTERVAL == 0
            or step == MAX_STEPS
        ):

            answer = generate_sample(
                model,
                tokenizer,
                device,
            )

            print()
            print("=" * 60)

            print(
                f"CHAT SAMPLE @ STEP "
                f"{step}"
            )

            print("=" * 60)

            print(
                "USER:"
            )

            print(
                "Hi! How are you today?"
            )

            print()
            print(
                "MINIGPT:"
            )

            print(answer)

            print("=" * 60)
            print()

        # ------------------------------------------
        # Checkpoint
        # ------------------------------------------

        if (
            step % CHECKPOINT_INTERVAL == 0
            or step == MAX_STEPS
        ):

            path = save_checkpoint(
                model,
                optimizer,
                step,
                last_validation_loss,
            )

            print(
                f"Chat checkpoint saved: "
                f"{path}"
            )

    print()
    print("=" * 60)
    print("CHAT FINE-TUNING COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()