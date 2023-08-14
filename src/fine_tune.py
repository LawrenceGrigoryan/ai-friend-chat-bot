"""
LLaMa2 QLoRA fine-tuning script
"""
import logging
from dataclasses import dataclass
from typing import Tuple, NoReturn

import torch
import peft
import transformers
import wandb
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk, Dataset

from fine_tuning_config import FineTuningConfig

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


def build_lora_model(
        model_name: str,
        device_map: str = "auto",
        load_in_8bit: bool = True,
        torch_dtype: torch.dtype = torch.float16,
        lora_rank: int = 8,
        lora_alpha: int = 32
        ) -> Tuple[peft.peft_model.PeftModelForCausalLM,
                   transformers.PreTrainedTokenizer]:
    """
    Build LoRA model

    Args:
        model_name (str): Base model name
        model_load_params (dict): Loading parameters
        lora_rank (int, optional): Lora rank value. Defaults to 8.
        lora_alpha (int, optional): Lora alpha value. Defaults to 32.

    Returns:
        Tuple[peft.peft_model.PeftModelForCausalLM, transformers.Tokenizer]: model and tokenizer
    """
    # Load model and tokenizer
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch_dtype
    )
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create PEFT model
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    logging.info(model.print_trainable_parameters())

    return model, tokenizer


def load_and_tokenize_data(
        tokenizer: transformers.PreTrainedTokenizer,
        train_data_path: str = "../data/train.hf/",
        eval_data_path: str = "../data/validation.hf/"
        ) -> Tuple[Dataset, Dataset]:
    """
    Load and tokenize datasets

    Args:
        tokenizer (transformers.Tokenizer): Model tokenizer
        train_data_path (str, optional): Path to train data. 
                                         Defaults to "../data/train.hf/".
        eval_data_path (str, optional): Path to validation data. 
                                       Defaults to "../data/validation.hf/".

    Returns:
        Tuple[Dataset, Dataset]: Tokenized train and validation datasets
    """
    # Load
    train_data = load_from_disk(train_data_path)
    eval_data = load_from_disk(eval_data_path)

    # Tokenize sequences
    train_data = train_data.map(lambda samples: tokenizer(samples["sample"]),
                                batched=True)
    eval_data = eval_data.map(lambda samples: tokenizer(samples["sample"]),
                              batched=True)

    # Shuffle data
    train_data = train_data.shuffle(seed=42)
    eval_data = eval_data.shuffle(seed=42)

    return train_data, eval_data


def train(config: dataclass) -> NoReturn:
    """
    Train LLM with LoRA

    Args:
        config (dataclass): Config dataclass for fine-tuning

    Returns:
        NoReturn
    """
    # Load model and tokenizer
    model, tokenizer = build_lora_model(
        model_name=config.model_name,
        device_map=config.device_map,
        load_in_8bit=config.load_in_8bit,
        torch_dtype=config.torch_dtype,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha
    )

    # Load data
    train_data, eval_data = load_and_tokenize_data(
        tokenizer=tokenizer,
        train_data_path=config.train_data_path,
        eval_data_path=config.eval_data_path
    )

    # Prepare training args
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        optim=config.optimizer,
        report_to="wandb" if config.use_wandb else None,
        fp16=True,
        logging_steps=1,
        output_dir=config.output_dir
    )

    # Set up WANDB
    wandb.init(project=config.project_name,
               name=config.run_name,
               tags=["llm", "lora", "instructions fine-tuning"],
               group="LLaMa")

    # Train
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer,
                                                                   mlm=False)
    )
    model.config.use_cache = False  # Re-enable during inference for speed
    trainer.train()

    # Save adaptor weights
    trainer.model.save_pretrained(config.output_dir + config.out_model_name)

    # Evaluate the model
    if config.evaluate:
        trainer.evaluate()


if __name__ == "__main__":
    train(FineTuningConfig)
