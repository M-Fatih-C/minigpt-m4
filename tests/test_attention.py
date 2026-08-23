import torch

from config import (
    HEAD_SIZE,
    N_EMBD,
)

from dataset import TokenDataset

from model.attention import (
    CausalSelfAttentionHead,
)

from model.embedding import (
    GPTEmbedding,
)


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():

    torch.manual_seed(42)

    device = get_device()

    print("=" * 60)
    print("MiniGPT-M4 Attention Test")
    print("=" * 60)

    print()
    print(f"Device    : {device}")
    print(f"Embedding : {N_EMBD}")
    print(f"Head size : {HEAD_SIZE}")

    # ----------------------------------------------
    # Dataset
    # ----------------------------------------------

    dataset = TokenDataset()

    x, _ = dataset.get_batch(
        split="train",
        device=device,
    )

    # Testi küçük tutuyoruz.
    x = x[:2, :128]

    print()
    print(
        "Token input shape:",
        x.shape,
    )

    # ----------------------------------------------
    # Embedding
    # ----------------------------------------------

    embedding = GPTEmbedding().to(
        device
    )

    embedded = embedding(x)

    print(
        "Embedding shape  :",
        embedded.shape,
    )

    assert embedded.shape == (
        2,
        128,
        N_EMBD,
    )

    # ----------------------------------------------
    # Attention
    # ----------------------------------------------

    attention = (
        CausalSelfAttentionHead()
        .to(device)
    )

    # Dropout test sırasında kapalı.
    attention.eval()

    with torch.no_grad():

        output = attention(
            embedded
        )

    print(
        "Attention shape  :",
        output.shape,
    )

    assert output.shape == (
        2,
        128,
        HEAD_SIZE,
    )

    print()
    print("Shape tests passed ✓")

    # ----------------------------------------------
    # CAUSALITY TEST
    # ----------------------------------------------

    print()
    print("=" * 60)
    print("CAUSAL MASK TEST")
    print("=" * 60)

    original_tokens = x.clone()

    modified_tokens = x.clone()

    # 64. pozisyondan SONRASINI tamamen değiştiriyoruz.
    #
    # İlk 64 token aynı kalacak.
    modified_tokens[:, 64:] = torch.randint(
        low=0,
        high=2048,
        size=modified_tokens[:, 64:].shape,
        device=device,
    )

    with torch.no_grad():

        original_embeddings = embedding(
            original_tokens
        )

        modified_embeddings = embedding(
            modified_tokens
        )

        original_output = attention(
            original_embeddings
        )

        modified_output = attention(
            modified_embeddings
        )

    # İlk 64 pozisyon gelecekteki değişikliklerden
    # etkilenmemeli.
    difference = (
        original_output[:, :64]
        - modified_output[:, :64]
    ).abs().max()

    difference_value = (
        difference.cpu().item()
    )

    print(
        f"Max past difference: "
        f"{difference_value:.10f}"
    )

    if difference_value > 1e-5:

        raise RuntimeError(
            "Causal mask failed!"
        )

    print(
        "Future tokens cannot affect "
        "past positions ✓"
    )

    # ----------------------------------------------
    # Future SHOULD change
    # ----------------------------------------------

    future_difference = (
        original_output[:, 64:]
        - modified_output[:, 64:]
    ).abs().mean()

    future_difference_value = (
        future_difference.cpu().item()
    )

    print(
        f"Future difference    : "
        f"{future_difference_value:.6f}"
    )

    print()
    print("=" * 60)
    print("ALL ATTENTION TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()