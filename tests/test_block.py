import torch

from config import (
    BATCH_SIZE,
    BLOCK_SIZE,
    N_EMBD,
)

from dataset import TokenDataset

from model.block import (
    TransformerBlock,
)

from model.embedding import (
    GPTEmbedding,
)


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def main():

    torch.manual_seed(42)

    device = get_device()

    print("=" * 60)
    print("MiniGPT-M4 Transformer Block Test")
    print("=" * 60)

    print()
    print(f"Device     : {device}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Block size : {BLOCK_SIZE}")
    print(f"Embedding  : {N_EMBD}")

    dataset = TokenDataset()

    x, _ = dataset.get_batch(
        "train",
        device=device,
    )

    # Testi hafif tutalım.
    x = x[:2, :128]

    embedding = GPTEmbedding().to(
        device
    )

    block = TransformerBlock().to(
        device
    )

    block.eval()

    print()
    print(
        f"Transformer block parameters: "
        f"{count_parameters(block):,}"
    )

    with torch.no_grad():

        embedded = embedding(x)

        output = block(
            embedded
        )

    print()
    print(
        "Input tokens shape :",
        x.shape,
    )

    print(
        "Embedding shape    :",
        embedded.shape,
    )

    print(
        "Block output shape :",
        output.shape,
    )

    assert output.shape == (
        2,
        128,
        N_EMBD,
    )

    # NaN / Inf kontrolü
    assert torch.isfinite(
        output
    ).all()

    print()
    print(
        "Shape and finite-value tests passed ✓"
    )

    # ------------------------------------------------
    # CAUSALITY TEST
    # ------------------------------------------------

    print()
    print("=" * 60)
    print("TRANSFORMER BLOCK CAUSALITY TEST")
    print("=" * 60)

    original = x.clone()
    modified = x.clone()

    modified[:, 64:] = torch.randint(
        low=0,
        high=2048,
        size=modified[:, 64:].shape,
        device=device,
    )

    with torch.no_grad():

        original_output = block(
            embedding(original)
        )

        modified_output = block(
            embedding(modified)
        )

    past_difference = (
        original_output[:, :64]
        - modified_output[:, :64]
    ).abs().max()

    future_difference = (
        original_output[:, 64:]
        - modified_output[:, 64:]
    ).abs().mean()

    print(
        f"Past difference   : "
        f"{past_difference.item():.10f}"
    )

    print(
        f"Future difference : "
        f"{future_difference.item():.6f}"
    )

    if past_difference.item() > 1e-5:

        raise RuntimeError(
            "Transformer block causality failed!"
        )

    print()
    print(
        "Transformer block causality passed ✓"
    )

    print()
    print("=" * 60)
    print("ALL BLOCK TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()