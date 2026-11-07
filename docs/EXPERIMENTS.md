# Experiments and evidence

The repository contains code, a serialized BPE tokenizer and two archived SFT
experiments. Training corpora, checkpoint weights and independent run logs are
excluded. Historical figures below were recorded by the author in the previous
README; they were not reproduced by the audit.

## TinyStories pretraining

| Quantity | Historical record / derivation |
|---|---|
| Prepared training corpus | 289,234 stories; approximately 250 MB; 115,706,628 tokens |
| Prepared validation corpus | 21,990 stories; 8,491,377 tokens |
| Described training configuration | 32 sequences × 256 tokens × 5,000 steps |
| Token exposure in that run | 40,960,000 token positions, derived from the configuration |
| Reported validation loss | 1.3553; unverified historical observation |
| Referenced checkpoint | `checkpoints/step_005000.pt`, excluded from Git |

Prepared corpus size and training exposure measure different things. Random
window sampling does not establish that every unique corpus token was visited.
Current configuration code is not proof that a run completed.

## Archived dialogue experiments

- `archive/dailydialog/`: flat dialogue streams with assistant-only targets.
  Random windows can detach a reply from its user context. Historical paths and
  external data/checkpoints are required.
- `archive/smoltalk_filtered/`: conversation-aware examples from
  `HuggingFaceTB/smol-smoltalk`, filtered to `smol-magpie-ultra-short`.
  Historical records list 7,336 training examples / 1,446,496 tokens / 836,518
  assistant targets, and 793 validation examples / 157,105 tokens / 92,293
  assistant targets. Verify these by rebuilding the exact data selection.
  Qualitative notes describe repetitive mathematical replies; lower loss alone
  does not establish broad conversational ability.

Role markers are text encoded by the existing tokenizer, rather than separately
reserved special-token IDs.

## Reproduction and limits

Record dataset revisions, selection parameters, tokenizer hash, dependency
versions, seed and configuration with new runs. Check dataset terms before
redistributing data or trained artefacts. The audit's synthetic tests validate
software contracts, not historical metrics or model usefulness.

No benchmark establishes factual reliability, general instruction following or
specialist suitability. The small model can generate incorrect or repetitive text.

## Roadmap

Broader raw-conversation SFT, fixed qualitative prompts, reproducible run manifests
and larger/context-scaled experiments remain future work. `benchmark_scale.py`
is not proof that a particular scale configuration was trained.
