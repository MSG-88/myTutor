"""Task builders that convert cleaned chunks to supervised examples."""
from typing import Iterable
from slm_finetune_framework.core.interfaces import RawChunk, TrainingExample, BaseTaskBuilder
from slm_finetune_framework.core.registry import TaskBuilderRegistry


class SummarizationTaskBuilder(BaseTaskBuilder):
    """Build simple summarization-style instruction examples."""

    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt or (
            "You are a domain expert. Summarize the key points from the context."
        )

    def build_examples(self, chunks: Iterable[RawChunk]) -> Iterable[TrainingExample]:
        for c in chunks:
            input_text = (
                f"{self.system_prompt}\n\n"
                f"Context:\n{c.content}\n\n"
                f"Task: Summarize the context in a concise way."
            )
            # Placeholder: in real system, fill target_text with real or synthetic labels
            target_text = "DUMMY_LABEL"  # Codex: mark for replacement later
            yield TrainingExample(
                input_text=input_text,
                target_text=target_text,
                metadata=c.metadata,
            )


TaskBuilderRegistry.register("summarization", SummarizationTaskBuilder)
