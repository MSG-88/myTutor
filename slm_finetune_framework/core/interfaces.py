"""Core interfaces for connectors, processing, and training artifacts."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Any, Optional


@dataclass
class RawChunk:
    """Represents a raw chunk of text extracted from a source."""

    id: str
    source_type: str  # "pdf", "postgres", "jira", "html", ...
    source_uri: str  # path/URL/connection-id
    metadata: Dict[str, Any]
    content: str  # extracted plain text


@dataclass
class TrainingExample:
    """Represents a single supervised training pair."""

    input_text: str
    target_text: str
    metadata: Dict[str, Any]


class BaseConnector(ABC):
    """Abstract base for all connectors."""

    @abstractmethod
    def list_resources(self) -> Iterable[str]:
        """Return resource identifiers (file paths, table names, URLs, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def load_resource(self, resource_id: str) -> Iterable[RawChunk]:
        """Extract text chunks for a single resource."""
        raise NotImplementedError


class BaseTaskBuilder(ABC):
    """Converts RawChunks -> TrainingExamples for a particular task type."""

    @abstractmethod
    def build_examples(self, chunks: Iterable[RawChunk]) -> Iterable[TrainingExample]:
        """Transform raw chunks into task-specific supervised examples."""
        raise NotImplementedError
