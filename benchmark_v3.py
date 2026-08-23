import argparse
import json
import time

import torch

from config_v3 import (
    BLOCK_SIZE,
    N_EMBD,
    N_HEAD,
    N_LAYER,
    VOCAB_SIZE,
)

from model_v3 import MiniGPTV3


def bytes_to_gb(value):
    if value is None:
        return None

    return value / (1024 ** 3)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is not available."
        )

    device = torch.device("mps")

    torch.manual_seed(42)

    torch.mps.empty_cache()

    print("=" * 88)
    print("MiniGPT-M4 V3 80M Benchmark")
    print("=" * 88)

    print()
    print(
        f"Device     : {device}"
    )

    print(
        f"Vocabulary : {VOCAB_SIZE:,}"
    )

    print(
        f"Context    : {BLOCK_SIZE}"
    )

    print(
        f"Embedding  : {N_EMBD}"
    )

    print(
        f"Heads      : {N_HEAD}"
    )

    print(
        f"Layers     : {N_LAYER}"
    )

    print(
        f"Batch      : {args.batch}"
    )

    print(
        f"Warmup     : {args.warmup}"
    )

    print(
        f"Steps      : {args.steps}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MiniGPTV3().to(
        device
    )

    model.train()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        f"Parameters : "
        f"{parameter_count:,}"
    )

    print(
        f"Parameters : "
        f"{parameter_count / 1e6:.4f}M"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(
            0.9,
            0.95,
        ),
        weight_decay=0.1,
    )

    # --------------------------------------------------------
    # Synthetic batch
    # --------------------------------------------------------

    x = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(
            args.batch,
            BLOCK_SIZE,
        ),
        dtype=torch.long,
        device=device,
    )

    y = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(
            args.batch,
            BLOCK_SIZE,
        ),
        dtype=torch.long,
        device=device,
    )

    def train_step():

        optimizer.zero_grad(
            set_to_none=True
        )

        _, loss = model(
            x,
            y,
        )

        loss.backward()

        gradient_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                model.parameters(),
                1.0,
            )
        )

        optimizer.step()

        return (
            loss.item(),
            float(gradient_norm),
        )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print()
    print(
        "Warmup..."
    )

    for _ in range(
        args.warmup
    ):
        train_step()

    torch.mps.synchronize()

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print(
        "Benchmark..."
    )

    losses = []
    gradients = []

    start = time.perf_counter()

    for _ in range(
        args.steps
    ):

        loss, grad = train_step()

        losses.append(
            loss
        )

        gradients.append(
            grad
        )

    torch.mps.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    total_tokens = (
        args.batch
        * BLOCK_SIZE
        * args.steps
    )

    tokens_per_second = (
        total_tokens
        / elapsed
    )

    milliseconds_per_step = (
        elapsed
        / args.steps
        * 1000
    )

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    allocated = None
    driver = None
    recommended = None

    if hasattr(
        torch.mps,
        "current_allocated_memory",
    ):

        allocated = (
            torch.mps
            .current_allocated_memory()
        )

    if hasattr(
        torch.mps,
        "driver_allocated_memory",
    ):

        driver = (
            torch.mps
            .driver_allocated_memory()
        )

    if hasattr(
        torch.mps,
        "recommended_max_memory",
    ):

        recommended = (
            torch.mps
            .recommended_max_memory()
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print()
    print("=" * 88)
    print("RESULT")
    print("=" * 88)

    print()
    print(
        f"Average loss      : "
        f"{sum(losses) / len(losses):.4f}"
    )

    print(
        f"Average grad norm : "
        f"{sum(gradients) / len(gradients):.4f}"
    )

    print(
        f"Time / step       : "
        f"{milliseconds_per_step:.1f} ms"
    )

    print(
        f"Tokens / second   : "
        f"{tokens_per_second:,.0f}"
    )

    if allocated is not None:

        print(
            f"MPS allocated     : "
            f"{bytes_to_gb(allocated):.2f} GB"
        )

    if driver is not None:

        print(
            f"Driver allocated  : "
            f"{bytes_to_gb(driver):.2f} GB"
        )

    if recommended is not None:

        print(
            f"Recommended max   : "
            f"{bytes_to_gb(recommended):.2f} GB"
        )

    print()
    print(
        "V3 benchmark complete ✓"
    )


if __name__ == "__main__":
    main()
