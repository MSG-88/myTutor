"""Utilities for breaking raw text into manageable pieces."""
from typing import Iterable, List
from slm_finetune_framework.core.interfaces import RawChunk


def simple_char_chunk(text: str, max_chars: int = 2000) -> List[str]:
    """Split text into roughly equal character-length segments."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        start = end
    return chunks


def chunk_raw_chunks(chunks: Iterable[RawChunk], max_chars: int = 2000) -> Iterable[RawChunk]:
    """Chunk RawChunk.content fields and propagate metadata."""
    for rc in chunks:
        splits = simple_char_chunk(rc.content, max_chars=max_chars)
        for idx, s in enumerate(splits):
            yield RawChunk(
                id=f"{rc.id}::chunk{idx}",
                source_type=rc.source_type,
                source_uri=rc.source_uri,
                metadata={**rc.metadata, "chunk_idx": idx},
                content=s,
            )
