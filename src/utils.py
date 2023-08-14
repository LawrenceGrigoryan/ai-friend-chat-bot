"""
Utility functions
"""
import logging
from typing import Union

import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline
)
from peft import PeftModel

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


def get_model(
        model_path: str,
        model_load_params: dict,
        adaptor_weights_path: str = Union[str, None]
        ) -> pipeline:
    """
    Get model pipeline for inference

    Args:
        model_path (str): LLM model path
        model_load_params (dict): model loading params
        adaptor_weights_path (str, optional): If give, load adaptor weights.
                                              Defaults to None.

    Returns:
        pipeline: model pipeline
    """
    logging.info("Loading model and tokenizer")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        **model_load_params,
    )
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    if adaptor_weights_path:
        logging.info("Loading adaptor weights")
        model = PeftModel.from_pretrained(
            model,
            adaptor_weights_path,
            torch_dtype=torch.float16
        )

    logging.info("Creating pipeline")
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return model_pipeline
