"""
QLoRA fine-tuning config dataclass
"""
from dataclasses import dataclass

import torch


@dataclass
class FineTuningConfig:
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    device_map: str = "auto"
    load_in_8bit: bool = True
    torch_dtype: torch.dtype = torch.float16
    lora_rank: int = 8
    lora_alpha: int = 32
    train_data_path: str = "../data/train.hf/"
    eval_data_path: str = "../data/validation.hf/"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 30
    max_steps: int = 200
    num_train_epochs: int = 1
    learning_rate: int = 5e-5
    weight_decay: int = 0.03
    optimizer: str = "adamw_bnb_8bit"
    evaluate: bool = False
    output_dir: str = "../models/"
    out_model_name: str = "llama-chat-7b-lora-friendly-dialogue"
    use_wandb: bool = True
    project_name: str = "llm-friend-chat-bot"
    run_name: str = "llm_lora_fine_tuning"
