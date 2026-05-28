import os
from collections.abc import Iterable, Iterator
from functools import lru_cache
import pickle
import regex as re
import numpy as np
from pathlib import Path

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

PAT_RE = re.compile(PAT)

BYTE_TOKENS = tuple(bytes([i]) for i in range(256))

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        cache_size=1000000
    ):
        """
        Construct a BPE tokenizer from a vocabulary, merge list, and optional
        special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        existing_tokens = set(self.vocab.values())

        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")
            if special_bytes not in existing_tokens:
                self.vocab[len(self.vocab)] = special_bytes
                existing_tokens.add(special_bytes)

        self.token_to_id = {b: i for i, b in self.vocab.items()}
        self.merge_ranks = {m: i for i, m in enumerate(self.merges)}
        self.special_token_bytes = {t: t.encode("utf-8") for t in self.special_tokens}
        self.special_token_ids = {
            t: self.token_to_id[self.special_token_bytes[t]]
            for t in self.special_tokens
        }

        if self.special_tokens:
            escaped = [re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)]
            self.special_re = re.compile("(" + "|".join(escaped) + ")")
        else:
            self.special_re = None

        # Bind an LRU-cached version per tokenizer instance.
        # Caching is usually the largest win because natural text repeats many pretokens.
        self._bpe_encode_bytes = lru_cache(maxsize=cache_size)(self._bpe_encode_bytes_uncached)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        Construct a Tokenizer from serialized vocab and merges files.
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)
    
    def _bpe_encode_bytes_uncached(self, pretoken_bytes: bytes) -> tuple[int, ...]:
        """
        Encode one regex pretoken, represented as raw UTF-8 bytes.

        This preserves the original algorithm:
        repeatedly choose the available merge with the lowest rank, then merge
        all occurrences of that pair.
        """
        parts = [BYTE_TOKENS[b] for b in pretoken_bytes]

        merge_ranks = self.merge_ranks
        token_to_id = self.token_to_id

        while len(parts) > 1:
            best_pair = None
            best_rank = None

            # Find the currently best-ranked adjacent pair.
            prev = parts[0]
            for cur in parts[1:]:
                pair = (prev, cur)
                rank = merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
                prev = cur

            if best_pair is None:
                break

            # Merge all non-overlapping occurrences of best_pair.
            merged = []
            i = 0
            last = len(parts) - 1
            left, right = best_pair

            while i < last:
                if parts[i] == left and parts[i + 1] == right:
                    merged.append(left + right)
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1

            if i == last:
                merged.append(parts[-1])

            parts = merged

        return tuple(token_to_id[token] for token in parts)

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of token IDs.
        """
        if self.special_re is not None:
            parts = self.special_re.split(text)
        else:
            parts = (text,)

        tokens: list[int] = []
        special_token_ids = self.special_token_ids
        bpe_encode_bytes = self._bpe_encode_bytes

        for part in parts:
            if not part:
                continue

            special_id = special_token_ids.get(part)
            if special_id is not None:
                tokens.append(special_id)
                continue

            for match in PAT_RE.finditer(part):
                tokens.extend(bpe_encode_bytes(match.group(0).encode("utf-8")))

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings, yielding token IDs one at a time.
        Useful for large files.
        """
        for elem in iterable:
            yield from self.encode(elem)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back into text.
        """
        b = b"".join(self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    def cache_info(self):
        """
        Inspect the BPE pretoken cache hit rate.
        """
        return self._bpe_encode_bytes.cache_info()

    def clear_cache(self):
        """
        Clear the BPE pretoken cache.
        """
        self._bpe_encode_bytes.cache_clear()


def tokenize_file_to_uint16(tokenizer: Tokenizer, input_path, output_path):
    """
    Tokenize a UTF-8 text file and save token ids as a .npy uint16 array.

    Note: uint16 is only safe when all token IDs are <= 65535.
    """
    with open(input_path, encoding="utf-8") as f:
        arr = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)

    np.save(output_path, arr)
    return arr



if __name__ == "__main__":
    repo_dir = Path(__file__).parent.parent
    out_dir = repo_dir / 'out'
    tinystories_tokenizer = Tokenizer.from_files(
        out_dir / "tinystories_bpe_vocab.pkl", out_dir / "tinystories_bpe_merges.pkl", ["<|endoftext|>"]
    )
    tokenize_file_to_uint16(
        tokenizer=tinystories_tokenizer,
        input_path=repo_dir / "data" / "TinyStoriesV2-GPT4-train.txt",
        output_path=out_dir / "tinystories_train_tokens.npy",
    )

    tokenize_file_to_uint16(
        tokenizer=tinystories_tokenizer,
        input_path=repo_dir / "data" / "TinyStoriesV2-GPT4-valid.txt",
        output_path=out_dir / "tinystories_valid_tokens.npy",
    )

    owt_tokenizer = Tokenizer.from_files(repo_dir / "owt_bpe_vocab.pkl", repo_dir / "owt_bpe_merges.pkl", [])
    tokenize_file_to_uint16(
        tokenizer=owt_tokenizer,
        input_path="data/owt_train.txt",
        output_path="out/owt_train_tokens.npy",
    )

    tokenize_file_to_uint16(
        tokenizer=owt_tokenizer,
        input_path="data/owt_valid.txt",
        output_path="out/owt_valid_tokens.npy",
    )
