import torch

from chat_dataset import ChatDataset


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():

    device = get_device()

    print("=" * 60)
    print("MiniGPT-M4 Chat Batch Test")
    print("=" * 60)

    print(
        f"Device: {device}"
    )

    dataset = ChatDataset(
        "train"
    )

    x, y = dataset.get_batch(
        device=device,
    )

    print()
    print(
        "X shape:",
        x.shape,
    )

    print(
        "Y shape:",
        y.shape,
    )

    learned = (
        y != -100
    ).sum().item()

    ignored = (
        y == -100
    ).sum().item()

    print()
    print(
        f"Learned targets : "
        f"{learned:,}"
    )

    print(
        f"Ignored targets : "
        f"{ignored:,}"
    )

    print(
        f"Learn ratio     : "
        f"{100 * learned / y.numel():.2f}%"
    )

    assert learned > 0
    assert ignored > 0

    print()
    print(
        "Assistant-only masking passed ✓"
    )


if __name__ == "__main__":
    main()