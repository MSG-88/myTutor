"""Connector for local and shared document files."""
import os
from typing import Iterable, List
from slm_finetune_framework.core.interfaces import BaseConnector, RawChunk
from slm_finetune_framework.core.registry import ConnectorRegistry


class DocumentConnector(BaseConnector):
    """Simple document connector that walks a directory tree."""

    def __init__(self, root_path: str, include_exts: List[str] | None = None):
        self.root_path = root_path
        self.include_exts = [e.lower() for e in include_exts] if include_exts else None

    def list_resources(self) -> Iterable[str]:
        for dirpath, _, filenames in os.walk(self.root_path):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if self.include_exts and ext not in self.include_exts:
                    continue
                yield os.path.join(dirpath, fname)

    def load_resource(self, resource_id: str) -> Iterable[RawChunk]:
        ext = os.path.splitext(resource_id)[1].lower()
        text = self._extract_text(resource_id, ext)
        if not text:
            return []
        yield RawChunk(
            id=resource_id,
            source_type=f"document:{ext}",
            source_uri=resource_id,
            metadata={"ext": ext},
            content=text,
        )

    def _extract_text(self, path: str, ext: str) -> str:
        # TODO: implement real handlers per type
        if ext == ".txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        # Add PDF, DOCX, PPTX, HTML extraction here
        return ""


ConnectorRegistry.register("documents", DocumentConnector)
