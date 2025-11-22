"""Model loading and LoRA application utilities."""
from dataclasses import dataclass
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


@dataclass
class ModelConfig:
    """Configuration for base model and LoRA hyperparameters."""

    base_model_path: str  # local model dir
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None  # ["q_proj", "v_proj"]
    max_length: int = 2048


class ModelManager:
    """Wrapper that owns tokenizer/model and applies LoRA adapters."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_path,
            device_map="auto",
            torch_dtype="auto",
        )

    def apply_lora(self):
        """Attach LoRA adapters to the base model in-place."""
        lora_config = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=self.cfg.target_modules or ["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)

    def save_adapter(self, output_dir: str):
        """Persist the adapter weights and tokenizer."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
