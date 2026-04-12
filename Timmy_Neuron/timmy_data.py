"""
timmy_data.py
Data Loading and Tokenization for Timmy Training

This module provides the data pipeline that feeds tokenized text to Timmy.
It handles three input sources:

    1. HuggingFace datasets (e.g. "roneneldan/TinyStories", "openwebtext")
    2. Local text files (one document per line, or raw text)
    3. Pre-tokenized tensor files (for resumed training)

The pipeline tokenizes text into fixed-length chunks of max_seq_len tokens,
batches them, and yields (token_ids,) tuples ready for TimmyModel.forward().

The tokenizer is configurable via TimmyConfig.tokenizer_id. Default is
meta-llama/Llama-3.2-1B, which uses the Llama 3 tokenizer with 128,256
vocabulary entries. Any HuggingFace tokenizer can be substituted.
"""

from __future__ import annotations

import os
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Optional, List, Iterator
from pathlib import Path


# =========================================================================
# Tokenizer Wrapper
# =========================================================================

def get_tokenizer(tokenizer_id: str = "meta-llama/Llama-3.2-1B"):
    """
    Load a HuggingFace tokenizer.

    Args:
        tokenizer_id: HuggingFace model ID or local path.

    Returns:
        A tokenizer object with encode() and decode() methods.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# =========================================================================
# Chunked Text Dataset (Map-style, for local files)
# =========================================================================

class ChunkedTextDataset(Dataset):
    """
    Tokenizes a text file (or list of text files) into fixed-length chunks.

    Each chunk is max_seq_len tokens. Documents are concatenated with an
    EOS token between them, then split into chunks. The last chunk is
    padded if shorter than max_seq_len.

    This is the standard "pack and chunk" approach used for language model
    training: no attention masking across document boundaries, just raw
    token sequences.
    """

    def __init__(
        self,
        text_paths: List[str],
        tokenizer_id: str = "meta-llama/Llama-3.2-1B",
        max_seq_len: int = 512,
        verbose: bool = True,
    ):
        """
        Args:
            text_paths: list of paths to text files. Each file is read as
                a single string and tokenized.
            tokenizer_id: HuggingFace tokenizer to use.
            max_seq_len: chunk length in tokens.
            verbose: print progress during tokenization.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        tokenizer = get_tokenizer(tokenizer_id)
        eos = tokenizer.eos_token_id

        # Tokenize all files into one long token sequence.
        all_tokens = []
        for path in text_paths:
            if verbose:
                print(f"  Tokenizing {path}...")
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
            all_tokens.append(eos)

        if verbose:
            print(f"  Total tokens: {len(all_tokens):,}")

        # Split into fixed-length chunks.
        self.chunks = []
        for i in range(0, len(all_tokens) - max_seq_len, max_seq_len):
            self.chunks.append(
                torch.tensor(all_tokens[i : i + max_seq_len], dtype=torch.long)
            )

        # Handle the last partial chunk (pad with eos).
        remainder = all_tokens[len(self.chunks) * max_seq_len :]
        if len(remainder) > 1:
            padded = remainder + [eos] * (max_seq_len - len(remainder))
            self.chunks.append(torch.tensor(padded[:max_seq_len], dtype=torch.long))

        if verbose:
            print(f"  Chunks: {len(self.chunks):,} x {max_seq_len} tokens")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.chunks[idx]


# =========================================================================
# Streaming HuggingFace Dataset (Iterable-style, for large corpora)
# =========================================================================

class StreamingHFDataset(IterableDataset):
    """
    Streams tokenized chunks from a HuggingFace dataset without loading
    the entire corpus into memory.

    Suitable for large datasets like OpenWebText (~40GB), C4, or
    RedPajama. The dataset is streamed from HuggingFace Hub and tokenized
    on the fly.

    Documents are concatenated into a running buffer. When the buffer
    reaches max_seq_len tokens, a chunk is yielded and the buffer is reset.
    """

    def __init__(
        self,
        dataset_name: str = "roneneldan/TinyStories",
        dataset_split: str = "train",
        text_column: str = "text",
        tokenizer_id: str = "meta-llama/Llama-3.2-1B",
        max_seq_len: int = 512,
    ):
        """
        Args:
            dataset_name: HuggingFace dataset identifier.
            dataset_split: which split to use ("train", "validation", etc.).
            text_column: name of the text field in the dataset.
            tokenizer_id: HuggingFace tokenizer to use.
            max_seq_len: chunk length in tokens.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.text_column = text_column
        self.tokenizer_id = tokenizer_id
        self.max_seq_len = max_seq_len

    def __iter__(self) -> Iterator[torch.Tensor]:
        from datasets import load_dataset

        tokenizer = get_tokenizer(self.tokenizer_id)
        eos = tokenizer.eos_token_id

        ds = load_dataset(
            self.dataset_name,
            split=self.dataset_split,
            streaming=True,
        )

        buffer = []
        for example in ds:
            text = example[self.text_column]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            buffer.append(eos)

            while len(buffer) >= self.max_seq_len:
                chunk = buffer[: self.max_seq_len]
                buffer = buffer[self.max_seq_len :]
                yield torch.tensor(chunk, dtype=torch.long)


# =========================================================================
# DataLoader Factory
# =========================================================================

def create_dataloader(
    source: str,
    tokenizer_id: str = "meta-llama/Llama-3.2-1B",
    max_seq_len: int = 512,
    batch_size: int = 2,
    num_workers: int = 2,
    shuffle: bool = True,
    text_column: str = "text",
    split: str = "train",
) -> DataLoader:
    """
    Create a DataLoader from either a HuggingFace dataset name or a local
    file/directory path.

    Args:
        source: either a HuggingFace dataset name (e.g. "roneneldan/TinyStories")
            or a path to a local text file or directory of text files.
        tokenizer_id: HuggingFace tokenizer to use.
        max_seq_len: chunk length in tokens.
        batch_size: batch size.
        num_workers: DataLoader worker processes.
        shuffle: whether to shuffle (only for map-style datasets).
        text_column: name of the text field (for HuggingFace datasets).
        split: dataset split (for HuggingFace datasets).

    Returns:
        A PyTorch DataLoader yielding batches of shape (batch_size, max_seq_len).

    Examples:
        # From HuggingFace:
        loader = create_dataloader("roneneldan/TinyStories", batch_size=4)

        # From a local text file:
        loader = create_dataloader("/data/corpus.txt", batch_size=4)

        # From a directory of text files:
        loader = create_dataloader("/data/texts/", batch_size=4)
    """
    source_path = Path(source)

    if source_path.exists():
        # Local file or directory.
        if source_path.is_dir():
            text_paths = sorted([
                str(p) for p in source_path.glob("*.txt")
            ])
            if not text_paths:
                raise FileNotFoundError(
                    f"No .txt files found in {source}"
                )
        else:
            text_paths = [str(source_path)]

        dataset = ChunkedTextDataset(
            text_paths=text_paths,
            tokenizer_id=tokenizer_id,
            max_seq_len=max_seq_len,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        # Assume HuggingFace dataset name.
        dataset = StreamingHFDataset(
            dataset_name=source,
            dataset_split=split,
            text_column=text_column,
            tokenizer_id=tokenizer_id,
            max_seq_len=max_seq_len,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
