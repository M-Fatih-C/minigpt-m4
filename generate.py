import argparse

import torch

from model.gpt import MiniGPT
from tokenizer.bpe import BPETokenizer


DEFAULT_CHECKPOINT = "checkpoints/step_005000.pt"
TOKENIZER_PATH = "tokenizer/tokenizer.json"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> MiniGPT:

    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = MiniGPT()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print(
        f"Checkpoint step: "
        f"{checkpoint.get('step', 'unknown')}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint.get('validation_loss', 'unknown')}"
    )

    return model


@torch.no_grad()
def generate_text(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:

    token_ids = tokenizer.encode(
        prompt
    )

    x = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    generated = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    return tokenizer.decode(
        generated[0]
        .cpu()
        .tolist()
    )


def main():

    parser = argparse.ArgumentParser(
        description="MiniGPT-M4 text generation"
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--prompt",
        default="Once upon a time, there was a little",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    device = get_device()

    print("=" * 60)
    print("MiniGPT-M4 Generator")
    print("=" * 60)

    print(f"Device      : {device}")
    print(f"Temperature : {args.temperature}")
    print(f"Top-k       : {args.top_k}")
    print(f"New tokens  : {args.tokens}")

    tokenizer = BPETokenizer.load(
        TOKENIZER_PATH
    )

    model = load_model(
        args.checkpoint,
        device,
    )

    print()
    print("=" * 60)
    print("PROMPT")
    print("=" * 60)
    print(args.prompt)

    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print()
    print("=" * 60)
    print("GENERATED")
    print("=" * 60)
    print()
    print(text)
    print()


if __name__ == "__main__":
    main()