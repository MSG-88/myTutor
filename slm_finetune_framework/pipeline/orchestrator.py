"""Pipeline orchestration for ingestion -> processing -> training."""
from typing import List
from slm_finetune_framework.core.interfaces import RawChunk, TrainingExample
from slm_finetune_framework.core.registry import ConnectorRegistry, TaskBuilderRegistry
from slm_finetune_framework.processing.cleaner import clean_chunks
from slm_finetune_framework.processing.chunker import chunk_raw_chunks
from slm_finetune_framework.training.model_manager import ModelConfig, ModelManager
from slm_finetune_framework.training.trainer import train_lora


class FinetunePipeline:
    """Run the end-to-end fine-tuning pipeline based on a config dict."""

    def __init__(self, config: dict):
        self.config = config

    def _init_connectors(self):
        connectors = []
        for conn_cfg in self.config["connectors"]:
            ctype = conn_cfg["type"]
            params = conn_cfg.get("params", {})
            connectors.append(ConnectorRegistry.create(ctype, **params))
        return connectors

    def _init_task_builder(self):
        tb_cfg = self.config["task_builder"]
        ttype = tb_cfg["type"]
        params = tb_cfg.get("params", {})
        return TaskBuilderRegistry.create(ttype, **params)

    def _init_model_manager(self):
        m_cfg = self.config["model"]
        model_cfg = ModelConfig(**m_cfg)
        return ModelManager(model_cfg)

    def run(self):
        connectors = self._init_connectors()
        task_builder = self._init_task_builder()
        model_mgr = self._init_model_manager()

        all_raw: List[RawChunk] = []
        for conn in connectors:
            for res_id in conn.list_resources():
                chunks = list(conn.load_resource(res_id))
                all_raw.extend(chunks)

        cleaned = list(clean_chunks(all_raw))
        chunked = list(chunk_raw_chunks(cleaned, max_chars=self.config["processing"]["max_chars"]))
        examples: List[TrainingExample] = list(task_builder.build_examples(chunked))

        train_cfg = self.config["training"]
        train_lora(
            model_mgr=model_mgr,
            examples=examples,
            output_dir=train_cfg["output_dir"],
            epochs=train_cfg["epochs"],
            batch_size=train_cfg["batch_size"],
            lr=train_cfg["lr"],
        )
