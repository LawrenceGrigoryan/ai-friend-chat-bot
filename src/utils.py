"""
Utility functions
"""
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline
)

from config import MODEL_PATH


def get_model() -> pipeline:
    """
    Loads pre-defined model into memory

    Returns:
        pipeline: transformers pipeline
    """
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH,
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             load_in_8bit=True)
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return model_pipeline
