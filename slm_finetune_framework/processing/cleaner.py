"""Simple text cleaning utilities."""
from typing import Iterable
from slm_finetune_framework.core.interfaces import RawChunk


def basic_clean(text: str) -> str:
    """Remove extraneous whitespace and trivial artifacts."""
    text = text.replace("\r", "\n")
    lines = [l.strip() for l in text.splitlines()]
    text = "\n".join(l for l in lines if l)
    text = " ".join(text.split())
    return text


def clean_chunks(chunks: Iterable[RawChunk]) -> Iterable[RawChunk]:
    """Yield cleaned versions of raw chunks, dropping empty results."""
    for c in chunks:
        cleaned = basic_clean(c.content)
        if not cleaned:
            continue
        c.content = cleaned
        yield c
