"""Lightweight supervised fine-tuning loop for LoRA adapters."""
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup, AdamW
from slm_finetune_framework.core.interfaces import TrainingExample
from .model_manager import ModelManager


class SupervisedDataset(Dataset):
    """PyTorch dataset that concatenates input and target text."""

    def __init__(self, examples: List[TrainingExample], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        text = ex.input_text + "\n\n" + ex.target_text
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": enc["attention_mask"][0],
        }


def train_lora(
    model_mgr: ModelManager,
    examples: List[TrainingExample],
    output_dir: str,
    epochs: int = 1,
    batch_size: int = 1,
    lr: float = 2e-4,
):
    """Train LoRA adapters on provided examples and save to output_dir."""
    model_mgr.apply_lora()
    model = model_mgr.model
    tokenizer = model_mgr.tokenizer
    max_length = model_mgr.cfg.max_length

    dataset = SupervisedDataset(examples, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(dataloader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(10, int(0.03 * num_training_steps)),
        num_training_steps=num_training_steps,
    )

    model.train()
    device = next(model.parameters()).device

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

    model_mgr.save_adapter(output_dir)
