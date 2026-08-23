import math

import torch

from config import (
    N_LAYER,
    VOCAB_SIZE,
)

from dataset import TokenDataset
from model.gpt import MiniGPT


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def count_parameters(
    model,
):

    return sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )


def main():

    torch.manual_seed(42)

    device = get_device()

    print("=" * 60)
    print("MiniGPT-M4 Full Model Test")
    print("=" * 60)

    print()
    print(
        f"Device        : {device}"
    )

    print(
        f"Transformer   : {N_LAYER} layers"
    )

    print(
        f"Vocabulary    : {VOCAB_SIZE:,}"
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = TokenDataset()

    x, y = dataset.get_batch(
        split="train",
        device=device,
    )

    # İlk forward testi için küçük batch.
    x = x[:2, :128]
    y = y[:2, :128]

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
        f"Parameters    : "
        f"{parameter_count:,}"
    )

    print(
        f"Parameters M  : "
        f"{parameter_count / 1_000_000:.3f}M"
    )

    # Weight tying gerçekten aynı
    # Parameter objesini kullanıyor mu?
    assert (
        model.lm_head.weight
        is
        model.embedding
        .token_embedding
        .weight
    )

    print(
        "Weight tying  : ✓"
    )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    model.eval()

    with torch.no_grad():

        logits, loss = model(
            x,
            y,
        )

    print()
    print("=" * 60)
    print("FORWARD TEST")
    print("=" * 60)

    print()
    print(
        "Input shape  :",
        x.shape,
    )

    print(
        "Target shape :",
        y.shape,
    )

    print(
        "Logits shape :",
        logits.shape,
    )

    assert logits.shape == (
        2,
        128,
        VOCAB_SIZE,
    )

    assert loss is not None

    assert torch.isfinite(
        loss
    )

    print()
    print(
        f"Initial loss : "
        f"{loss.item():.4f}"
    )

    # --------------------------------------------------
    # Random-model sanity check
    # --------------------------------------------------

    random_baseline = math.log(
        VOCAB_SIZE
    )

    print(
        f"Random baseline: "
        f"{random_baseline:.4f}"
    )

    difference = abs(
        loss.item()
        - random_baseline
    )

    print(
        f"Difference     : "
        f"{difference:.4f}"
    )

    # Random initialize edilmiş modelin loss'u
    # yaklaşık ln(vocab_size) olmalı.
    if difference > 1.0:

        raise RuntimeError(
            "Initial loss is suspiciously "
            "far from random baseline."
        )

    print()
    print(
        "Initial loss sanity check passed ✓"
    )

    # --------------------------------------------------
    # Finite logits
    # --------------------------------------------------

    assert torch.isfinite(
        logits
    ).all()

    print(
        "Finite logits test passed ✓"
    )

    print()
    print("=" * 60)
    print("FULL MINIGPT FORWARD PASS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()