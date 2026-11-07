"""Small CPU regressions requiring no corpus, downloads or trained checkpoint."""
import os
import tempfile
import unittest
from pathlib import Path

# A small configuration keeps contract tests practical on CPU-only CI.
os.environ.update(
    MINIGPT_N_EMBD="24", MINIGPT_N_HEAD="3", MINIGPT_N_LAYER="2",
    MINIGPT_BLOCK_SIZE="16", MINIGPT_DROPOUT="0",
)

import numpy as np
import torch
import torch.nn.functional as F

from dataset import TokenDataset
from model.gpt import MiniGPT
from tokenizer.bpe import BPETokenizer


class OfflineContracts(unittest.TestCase):
    def test_tokenizer_preserves_unicode_and_whitespace(self):
        tokenizer = BPETokenizer.load("tokenizer/tokenizer.json")
        for text in ("", "hello  world\n", "Türkçe: İığüşöç — 🌍"):
            with self.subTest(text=text):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_future_tokens_do_not_change_prefix_logits(self):
        torch.manual_seed(42)
        model = MiniGPT().eval()
        original = torch.randint(0, 256, (1, 12))
        changed = original.clone()
        changed[:, 6:] = (changed[:, 6:] + 17) % 256
        with torch.no_grad():
            before, _ = model(original)
            after, _ = model(changed)
        torch.testing.assert_close(before[:, :6], after[:, :6])

    def test_weight_tying_and_response_only_loss(self):
        model = MiniGPT().eval()
        self.assertIs(model.lm_head.weight, model.embedding.token_embedding.weight)
        tokens = torch.tensor([[1, 2, 3, 4]])
        targets = torch.tensor([[-100, -100, 4, 5]])
        logits, loss = model(tokens, targets)
        expected = F.cross_entropy(logits[0, 2:], targets[0, 2:])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNotNone(model.embedding.token_embedding.weight.grad)

    def test_dataset_shift_and_unknown_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.bin"
            np.arange(128, dtype=np.uint16).tofile(path)
            dataset = TokenDataset(str(path), str(path))
            x, y = dataset.get_batch("train", torch.device("cpu"), 3, 8)
            self.assertEqual(tuple(x.shape), (3, 8))
            self.assertEqual(x.dtype, torch.int64)
            torch.testing.assert_close(y, x + 1)
            with self.assertRaisesRegex(ValueError, "Unknown split"):
                dataset.get_batch("not-a-split", torch.device("cpu"), 3, 8)


if __name__ == "__main__":
    unittest.main()
