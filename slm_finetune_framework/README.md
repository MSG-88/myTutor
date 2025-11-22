# slm_finetune_framework

A lightweight, config-driven framework for ingesting data from diverse sources and fine-tuning small language models with LoRA/QLoRA adapters.

## Features
- Connector layer for documents, databases, and server-based sources with registries for easy extension.
- Processing utilities for cleaning and chunking text data.
- Task builders for constructing supervised datasets (currently summarization).
- Training utilities to apply LoRA adapters on Hugging Face causal language models.
- YAML-driven orchestration pipeline for repeatable workflows.

## Quickstart
1. Create or edit a configuration file (see `config/example_config.yaml`).
2. Run the pipeline:
   ```bash
   python -m slm_finetune_framework.main --config slm_finetune_framework/config/example_config.yaml
   ```
   or simply run the module directly:
   ```bash
   python slm_finetune_framework/main.py --config slm_finetune_framework/config/example_config.yaml
   ```

> Note: The example config uses placeholder paths/credentials. Update them for your environment and ensure the base model is available locally.

## Extending
- Add new connectors by subclassing `BaseConnector` and registering via `ConnectorRegistry.register()`.
- Add new tasks by subclassing `BaseTaskBuilder` and registering via `TaskBuilderRegistry.register()`.
- Expand processing steps with additional cleaners or chunking strategies.

## Future Improvements
- Rich document parsers (PDF/DOCX/PPTX/HTML), media transcription, and server APIs.
- Synthetic label generation, evaluation harnesses, and experiment tracking.
- Serving endpoints (e.g., FastAPI) for trained adapters.
