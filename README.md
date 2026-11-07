# MiniGPT-M4

An educational decoder-only Transformer implemented in PyTorch, with a custom
byte-level BPE tokenizer and local Apple Silicon training experiments. This is
an experimental model-training project. Pretrained weights and training data
are not included.

## Architecture

Token and learned position embeddings feed pre-LayerNorm Transformer blocks
with causal multi-head self-attention, residual connections and GELU feed-forward
layers. The vocabulary projection shares its weights with the token embedding.
Training supports `-100` target masking for assistant-only SFT loss.

| Default configuration | Value |
|---|---:|
| Vocabulary / context | 2,048 / 256 tokens |
| Embedding / attention heads | 384 / 6 |
| Transformer layers / FFN width | 8 / 1,536 |
| Dropout | 0.1 |

`config.py` defines current defaults and `MINIGPT_*` size overrides. MPS is used
when available; CPU is a fallback. Data and checkpoints must match the tokenizer
and model configuration.

## Repository map

- `model/`: embeddings, causal attention, blocks and GPT implementation.
- `tokenizer/`: BPE implementation and serialized tokenizer.
- `dataset.py`: `uint16` token files and shifted next-token batches.
- `train.py`, `generate.py`: pretraining and checkpoint-based generation.
- `scripts/`: corpus preparation, tokenizer training and token serialization.
- `archive/`: separate DailyDialog and filtered SmolTalk SFT experiments.
- `tests/`: data-dependent diagnostic modules and offline contract tests.

## Setup

Use Python 3.12 in a virtual environment. Apple Silicon/MPS is useful for training;
the offline tests run on CPU.

```bash
git clone https://github.com/M-Fatih-C/minigpt-m4.git
cd minigpt-m4
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

`requirements.txt` declares broad ranges. `requirements-lock.txt` is a recorded
environment snapshot, not a resolver-maintained lockfile.

## Training and generation

These commands download external datasets and can start substantial computation.
Review dataset terms, available storage and configuration first.

```bash
python -m scripts.prepare_data
python -m scripts.train_tokenizer
python -m scripts.prepare_tokens
python train.py
python generate.py --help
```

Preparation writes local text/binary files under `data/`; training writes excluded
checkpoints. Generation needs a compatible checkpoint. Load only checkpoints you
trust. A clean clone does not contain trained model responses.

## Testing

The small CPU contract suite needs only PyTorch and NumPy and creates synthetic
temporary data:

```bash
python -m unittest tests.test_offline_contracts -v
```

It checks causal prefix independence, weight sharing and masked loss, Unicode
tokenizer roundtrips, and batch alignment. The workflow runs this suite; no passing
badge is asserted here. Original diagnostic commands need prepared token files:

```bash
python -m tests.test_attention
python -m tests.test_block
python -m tests.test_gpt
python -m tests.test_batch
```

The audit initially found those four commands failed on missing data; all four
completed using explicitly synthetic token files in a separate copy. This does
not reproduce pretraining or SFT quality.

## Experiments and limitations

[Experiment notes](docs/EXPERIMENTS.md) distinguish the historically reported
115.7M prepared training tokens from 40.96M token positions processed in the
described run. Loss and SFT counts are labelled by their evidence level. Broader
SFT and model scaling remain roadmap items.

The model is experimental and not validated for reliable factual answers or
production use. Keep private text, datasets and checkpoints out of commits;
excluded files are not automatically anonymized. External datasets retain their
own usage terms.

## License and author

No license file is included; no new license or redistribution grant is asserted
here. Author: [Muhammet Fatih Çetintaş](https://github.com/M-Fatih-C).
