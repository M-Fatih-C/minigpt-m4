import json
import re

from collections import Counter
from functools import lru_cache
from pathlib import Path


# Metni whitespace ve whitespace olmayan parçalar halinde ayırıyoruz.
# Böylece "hello world\n" gibi bir metin:
#
# ["hello", " ", "world", "\n"]
#
# şeklinde parçalanıyor.
PRETOKEN_PATTERN = re.compile(r"\s+|[^\s]+")


def merge_pair(
    sequence: list[int],
    pair: tuple[int, int],
    new_token_id: int,
) -> list[int]:
    """
    Sequence içerisinde verilen token çiftini
    soldan sağa non-overlapping şekilde birleştirir.

    Örnek:

    sequence = [10, 20, 10, 20]
    pair     = (10, 20)
    new_id   = 256

    sonuç:
    [256, 256]
    """

    result = []

    i = 0

    while i < len(sequence):

        if (
            i < len(sequence) - 1
            and sequence[i] == pair[0]
            and sequence[i + 1] == pair[1]
        ):
            result.append(new_token_id)
            i += 2

        else:
            result.append(sequence[i])
            i += 1

    return result


class BPETokenizer:

    BASE_VOCAB_SIZE = 256

    def __init__(
        self,
        merges: list[tuple[int, int, int]] | None = None,
    ):
        self.merges = merges or []

        # pair -> (merge rank, new token id)
        #
        # Düşük rank = daha önce öğrenilmiş merge.
        self.merge_ranks = {
            (left, right): (rank, new_id)
            for rank, (left, right, new_id)
            in enumerate(self.merges)
        }

        self.id_to_bytes = self._build_token_bytes()

    @property
    def vocab_size(self) -> int:
        return self.BASE_VOCAB_SIZE + len(self.merges)

    def _build_token_bytes(self) -> dict[int, bytes]:
        """
        Her token ID'nin hangi byte dizisini temsil ettiğini oluşturur.

        İlk 256 token doğrudan byte değerleridir.
        """

        token_bytes = {
            i: bytes([i])
            for i in range(256)
        }

        for left, right, new_id in self.merges:

            token_bytes[new_id] = (
                token_bytes[left]
                + token_bytes[right]
            )

        return token_bytes

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 2048,
        min_pair_frequency: int = 2,
        log_every: int = 100,
    ) -> "BPETokenizer":

        if vocab_size < 256:
            raise ValueError(
                "vocab_size must be at least 256."
            )

        print("Pretokenizing training corpus...")

        pieces = PRETOKEN_PATTERN.findall(text)

        piece_counts = Counter(pieces)

        print(f"Text characters : {len(text):,}")
        print(f"Pretokens       : {len(pieces):,}")
        print(f"Unique pretokens: {len(piece_counts):,}")
        print()

        # Başlangıçta her pretoken byte'lardan oluşuyor.
        sequences = {
            piece: list(piece.encode("utf-8"))
            for piece in piece_counts
        }

        merges: list[tuple[int, int, int]] = []

        token_bytes = {
            i: bytes([i])
            for i in range(256)
        }

        total_merges = vocab_size - 256

        print(
            f"Training {total_merges:,} BPE merges..."
        )

        for merge_index in range(total_merges):

            pair_counts: Counter[
                tuple[int, int]
            ] = Counter()

            # Corpus içerisindeki bütün adjacent token
            # çiftlerini sayıyoruz.
            #
            # Aynı kelime 500 kez geçiyorsa pair count'a
            # 500 ağırlıkla katkıda bulunuyor.
            for piece, frequency in piece_counts.items():

                sequence = sequences[piece]

                if len(sequence) < 2:
                    continue

                for i in range(len(sequence) - 1):

                    pair = (
                        sequence[i],
                        sequence[i + 1],
                    )

                    pair_counts[pair] += frequency

            if not pair_counts:
                print(
                    "\nNo token pairs left. "
                    "Stopping tokenizer training."
                )
                break

            best_pair, best_frequency = (
                pair_counts.most_common(1)[0]
            )

            if best_frequency < min_pair_frequency:
                print(
                    "\nPair frequency became too low. "
                    "Stopping tokenizer training."
                )
                break

            new_token_id = (
                cls.BASE_VOCAB_SIZE
                + len(merges)
            )

            left, right = best_pair

            merges.append(
                (
                    left,
                    right,
                    new_token_id,
                )
            )

            token_bytes[new_token_id] = (
                token_bytes[left]
                + token_bytes[right]
            )

            # Öğrendiğimiz merge kuralını corpus üzerindeki
            # tüm pretokens'a uyguluyoruz.
            for piece in sequences:

                sequences[piece] = merge_pair(
                    sequences[piece],
                    best_pair,
                    new_token_id,
                )

            current_merge = merge_index + 1

            if (
                current_merge == 1
                or current_merge % log_every == 0
                or current_merge == total_merges
            ):

                merged_bytes = (
                    token_bytes[new_token_id]
                )

                display = merged_bytes.decode(
                    "utf-8",
                    errors="replace",
                )

                print(
                    f"Merge "
                    f"{current_merge:>4}/{total_merges}"
                    f" | frequency={best_frequency:>8,}"
                    f" | token={repr(display)}"
                )

        tokenizer = cls(merges)

        print()
        print("Tokenizer training complete.")
        print(
            f"Final vocabulary size: "
            f"{tokenizer.vocab_size:,}"
        )

        return tokenizer

    @lru_cache(maxsize=100_000)
    def _encode_piece(
        self,
        piece: str,
    ) -> tuple[int, ...]:

        sequence = list(
            piece.encode("utf-8")
        )

        while len(sequence) >= 2:

            best_pair = None
            best_rank = None
            best_new_id = None

            # Sequence içerisinde mevcut BPE merge
            # kurallarından hangisinin en yüksek
            # önceliğe sahip olduğunu buluyoruz.
            for i in range(len(sequence) - 1):

                pair = (
                    sequence[i],
                    sequence[i + 1],
                )

                merge_info = (
                    self.merge_ranks.get(pair)
                )

                if merge_info is None:
                    continue

                rank, new_id = merge_info

                if (
                    best_rank is None
                    or rank < best_rank
                ):
                    best_pair = pair
                    best_rank = rank
                    best_new_id = new_id

            # Bu piece üzerinde uygulanabilecek
            # başka merge yok.
            if best_pair is None:
                break

            sequence = merge_pair(
                sequence,
                best_pair,
                best_new_id,
            )

        return tuple(sequence)

    def encode(
        self,
        text: str,
    ) -> list[int]:

        tokens: list[int] = []

        for piece in PRETOKEN_PATTERN.findall(text):

            tokens.extend(
                self._encode_piece(piece)
            )

        return tokens

    def decode(
        self,
        token_ids: list[int],
    ) -> str:

        data = b"".join(
            self.id_to_bytes[token_id]
            for token_id in token_ids
        )

        return data.decode(
            "utf-8",
            errors="replace",
        )

    def save(
        self,
        path: str | Path,
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "version": 1,
            "type": "byte_bpe",
            "vocab_size": self.vocab_size,
            "merges": [
                [left, right, new_id]
                for left, right, new_id
                in self.merges
            ],
        }

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                indent=2,
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "BPETokenizer":

        path = Path(path)

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            payload = json.load(file)

        merges = [
            (
                int(left),
                int(right),
                int(new_id),
            )
            for left, right, new_id
            in payload["merges"]
        ]

        return cls(merges)