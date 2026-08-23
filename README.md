# MiniGPT-M4

MiniGPT-M4 is a small decoder-only GPT language model implemented from scratch in PyTorch and trained locally on Apple Silicon.

The purpose of the project is not to wrap an existing pretrained LLM, but to build and understand the complete language-model pipeline:

- byte-level BPE tokenization
- token embeddings
- learned positional embeddings
- causal self-attention
- multi-head attention
- transformer blocks
- autoregressive language modeling
- pretraining
- supervised fine-tuning
- response-only loss masking
- conversation-aware training
- local inference on Apple MPS

No pretrained model weights are used.

---

## Hardware

Development and training currently run locally on:

- Apple Mac mini
- Apple M4
- PyTorch MPS backend

---

## Current Model

| Configuration | Value |
|---|---:|
| Architecture | Decoder-only Transformer |
| Parameters | ~15.07M |
| Vocabulary | 2,048 |
| Context length | 256 |
| Embedding dimension | 384 |
| Transformer layers | 8 |
| Attention heads | 6 |
| Head size | 64 |
| FFN hidden dimension | 1,536 |
| Dropout | 0.1 |
| Activation | GELU |
| Normalization | Pre-LayerNorm |
| Position embeddings | Learned |
| Weight tying | Token embedding / LM head |

---

## Architecture

```text
Token IDs
   │
   ├── Token Embedding
   │
   └── Position Embedding
            │
            ▼
    Transformer Block × 8
            │
            ├── LayerNorm
            ├── Causal Multi-Head Attention
            ├── Residual Connection
            ├── LayerNorm
            ├── Feed Forward Network
            └── Residual Connection
            │
            ▼
       Final LayerNorm
            │
            ▼
          LM Head
            │
            ▼
          Logits

The causal attention mask prevents each token from attending to future tokens.
Tokenizer
The project uses a custom byte-level BPE tokenizer.
Configuration:
Base byte vocabulary : 256
BPE merges           : 1792
Final vocabulary     : 2048
Tokenizer implementation:
tokenizer/bpe.py
Serialized tokenizer:
tokenizer/tokenizer.json
TinyStories Pretraining
The first language-model pretraining experiment used TinyStories.
Prepared training corpus:
Train stories      : 289,234
Train text         : ~250 MB
Train tokens       : 115,706,628

Validation stories : 21,990
Validation tokens  : 8,491,377
Training configuration:
Batch size       : 32
Context          : 256
Tokens / batch   : 8192
Steps            : 5000
Token exposure   : 40,960,000

Peak LR          : 3e-4
Optimizer        : AdamW
Gradient clip    : 1.0
Scheduler        : Warmup + Cosine
Final validation loss:
1.3553
The clean pretrained baseline is:
checkpoints/step_005000.pt
Checkpoint files are intentionally excluded from Git.
Dialogue Experiments
DailyDialog
The first chat fine-tuning experiment used DailyDialog.
A response-only objective was implemented using:
<|USER|>
...
<|END_OF_TURN|>

<|ASSISTANT|>
...
<|END_OF_TURN|>
Only assistant response tokens contributed to the supervised loss.
The experiment revealed an important issue: randomly sampling windows from one flat dialogue token stream can separate an assistant response from the user message it is supposed to answer.
The experiment is preserved under:
archive/dailydialog/
Filtered SmolTalk Experiment
The second SFT experiment introduced conversation-aware training examples.
Instead of sampling arbitrary windows, each assistant response was associated with its corresponding user context.
Prepared data:
Train examples       : 7,336
Train tokens         : 1,446,496
Assistant targets    : 836,518

Validation examples  : 793
Validation tokens    : 157,105
Assistant targets    : 92,293
The training pipeline worked correctly and validation loss improved significantly.
However, qualitative inspection showed that the selected subset was disproportionately dominated by:
mathematics
short calculation tasks
problem solving
task-oriented Q&A
The model consequently learned repetitive patterns related to numbers, questions and calculations instead of developing broad conversational behavior.
This experiment is preserved under:
archive/smoltalk_filtered/
Important Lessons
The experiments have demonstrated several important properties of small language-model training.
Validation loss is not enough
A lower validation loss does not necessarily mean better conversational behavior.
Qualitative generation tests are required alongside validation metrics.
Conversation boundaries matter
Assistant responses must be trained with the user message they actually answer.
Dataset distribution matters
For a small model, a narrow training distribution can dominate behavior very quickly.
Context length is not the current bottleneck
Increasing the context window from 256 to 512 or 1024 will not solve weak prompt-response learning.
The model and training data must work correctly first.
Current Direction
The next experiment will move away from aggressively filtered conversational subsets.
The new pipeline will use a much broader corpus with minimal structural preprocessing:
Raw conversation dataset
          │
          ▼
Structural validation
          │
          ├── valid roles
          ├── non-empty messages
          ├── valid conversation ordering
          └── conversation boundaries
          │
          ▼
Conversation-aware examples
          │
          ▼
Language-model training
Content categories such as:
general conversation
coding
mathematics
factual questions
reasoning
roleplay
writing
will not be aggressively removed.
The goal is to preserve the natural distribution of the training corpus.
Planned Scaling
After validating the broader raw-data training approach, larger model configurations will be tested.
Potential configurations include:
~30M parameters
~50M parameters
~80M parameters
Later experiments may also include:
512-token context
larger vocabulary
longer pretraining
larger embedding dimensions
additional transformer layers
Scaling will be benchmarked against Apple M4 memory usage, training throughput and generation quality.
Repository Structure
minigpt-m4/
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── .gitignore
│
├── config.py
├── dataset.py
├── train.py
├── generate.py
│
├── model/
│   ├── attention.py
│   ├── block.py
│   ├── embedding.py
│   ├── feedforward.py
│   └── gpt.py
│
├── tokenizer/
│   ├── bpe.py
│   └── tokenizer.json
│
├── scripts/
│
├── tests/
│
├── archive/
│   ├── dailydialog/
│   └── smoltalk_filtered/
│
├── data/
└── checkpoints/
The following directories are intentionally excluded from Git:
.venv/
data/
checkpoints/
chat_checkpoints*/
Setup
Create a virtual environment:
python3 -m venv .venv
source .venv/bin/activate
Install dependencies:
pip install -r requirements.txt
Check MPS:
import torch

print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())
Tests
Run tests from the repository root:
python -m tests.test_attention
python -m tests.test_block
python -m tests.test_gpt
python -m tests.test_batch
Pretraining
Prepare data:
python -m scripts.prepare_data
Train tokenizer:
python -m scripts.train_tokenizer
Prepare binary token data:
python -m scripts.prepare_tokens
Train MiniGPT:
python train.py
Generate text:
python generate.py
Roadmap
 Byte-level BPE tokenizer
 Decoder-only GPT
 Causal self-attention
 Multi-head attention
 Transformer blocks
 TinyStories pretraining
 Apple MPS training
 Autoregressive generation
 Response-only supervised loss
 DailyDialog experiment
 Conversation-aware SFT
 Filtered SmolTalk experiment
 Broad raw conversation dataset
 Raw-data SFT experiment
 Standard evaluation prompt suite
 Larger model benchmark
 512-token context experiment
 Improved special-token handling
 Published pretrained checkpoints
Status
MiniGPT-M4 is an experimental educational language-model project.
The current objective is to understand and improve the complete model-training pipeline before aggressively increasing model scale.
EOF