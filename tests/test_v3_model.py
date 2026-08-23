import torch

from config_v3 import (
    BLOCK_SIZE,
    EXPECTED_PARAMETER_COUNT,
    N_EMBD,
    N_HEAD,
    N_LAYER,
    VOCAB_SIZE,
)

from model_v3 import MiniGPTV3


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():

    torch.manual_seed(42)

    device = get_device()

    print("=" * 60)
    print("MiniGPT-M4 V3 80M Test")
    print("=" * 60)

    print()
    print(
        f"Device      : {device}"
    )

    print(
        f"Vocabulary  : {VOCAB_SIZE:,}"
    )

    print(
        f"Context     : {BLOCK_SIZE}"
    )

    print(
        f"Embedding   : {N_EMBD}"
    )

    print(
        f"Heads       : {N_HEAD}"
    )

    print(
        f"Layers      : {N_LAYER}"
    )

    model = MiniGPTV3().to(
        device
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        f"Parameters  : "
        f"{parameter_count:,}"
    )

    print(
        f"Parameters M: "
        f"{parameter_count / 1e6:.4f}M"
    )

    assert (
        parameter_count
        == EXPECTED_PARAMETER_COUNT
    ), (
        parameter_count,
        EXPECTED_PARAMETER_COUNT,
    )

    print(
        "Parameter count ✓"
    )

    # --------------------------------------------------------
    # Weight tying
    # --------------------------------------------------------

    assert (
        model.lm_head.weight.data_ptr()
        ==
        model.token_embedding.weight.data_ptr()
    )

    print(
        "Weight tying   ✓"
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    batch_size = 2
    sequence_length = 128

    x = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(
            batch_size,
            sequence_length,
        ),
        device=device,
    )

    y = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(
            batch_size,
            sequence_length,
        ),
        device=device,
    )

    logits, loss = model(
        x,
        y,
    )

    print()
    print("=" * 60)
    print("FORWARD")
    print("=" * 60)

    print()
    print(
        f"Input  : {x.shape}"
    )

    print(
        f"Logits : {logits.shape}"
    )

    print(
        f"Loss   : {loss.item():.4f}"
    )

    assert logits.shape == (
        batch_size,
        sequence_length,
        VOCAB_SIZE,
    )

    assert torch.isfinite(
        logits
    ).all()

    assert torch.isfinite(
        loss
    )

    print(
        "Forward pass    ✓"
    )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    loss.backward()

    gradients = [
        p.grad
        for p in model.parameters()
        if p.grad is not None
    ]

    assert gradients

    assert all(
        torch.isfinite(g).all()
        for g in gradients
    )

    print(
        "Backward pass   ✓"
    )

    # --------------------------------------------------------
    # Causality test
    # --------------------------------------------------------

    model.eval()

    test_length = 64

    original = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(1, test_length),
        device=device,
    )

    modified = original.clone()

    modified[
        :,
        test_length // 2:
    ] = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(
            1,
            test_length // 2,
        ),
        device=device,
    )

    with torch.no_grad():

        logits_a, _ = model(
            original
        )

        logits_b, _ = model(
            modified
        )

    past_difference = (
        logits_a[
            :,
            :test_length // 2,
            :
        ]
        -
        logits_b[
            :,
            :test_length // 2,
            :
        ]
    ).abs().max().item()

    print()
    print(
        f"Causality diff : "
        f"{past_difference:.10f}"
    )

    assert past_difference < 1e-5

    print(
        "Causality       ✓"
    )

    print()
    print("=" * 60)
    print(
        "MINIGPT-M4 V3 80M PASSED ✓"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
