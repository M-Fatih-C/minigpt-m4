import torch

from config import (
    BATCH_SIZE,
    BLOCK_SIZE,
)

from dataset import TokenDataset

from tokenizer.bpe import BPETokenizer


def get_device() -> torch.device:

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():

    device = get_device()

    print("=" * 60)
    print("MiniGPT-M4 Batch Test")
    print("=" * 60)

    print()
    print(f"Device     : {device}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Block size : {BLOCK_SIZE}")

    print()

    dataset = TokenDataset()

    x, y = dataset.get_batch(
        split="train",
        device=device,
    )

    print()
    print("=" * 60)
    print("TENSOR TEST")
    print("=" * 60)

    print("X shape :", x.shape)
    print("Y shape :", y.shape)

    print("X dtype :", x.dtype)
    print("Y dtype :", y.dtype)

    print("X device:", x.device)
    print("Y device:", y.device)

    print()
    print("=" * 60)
    print("SHIFT TEST")
    print("=" * 60)

    print("X first 20:")
    print(
        x[0, :20].tolist()
    )

    print()
    print("Y first 20:")
    print(
        y[0, :20].tolist()
    )

    # Kritik test:
    #
    # X:
    # [A B C D]
    #
    # Y:
    # [B C D E]
    #
    assert torch.equal(
        x[:, 1:],
        y[:, :-1],
    )

    print()
    print("X/Y shift test passed ✓")

    print()
    print("=" * 60)
    print("TEXT TEST")
    print("=" * 60)

    tokenizer = BPETokenizer.load(
        "tokenizer/tokenizer.json"
    )

    sample_x = tokenizer.decode(
        x[0].cpu().tolist()
    )

    sample_y = tokenizer.decode(
        y[0].cpu().tolist()
    )

    print()
    print("--- X ---")
    print(sample_x)

    print()
    print("--- Y ---")
    print(sample_y)

    print()
    print("Batch pipeline passed ✓")


if __name__ == "__main__":
    main()