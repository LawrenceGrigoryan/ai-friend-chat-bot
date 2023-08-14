"""
Utility functions
"""
import os
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

HF_AUTH_TOKEN = os.environ["HF_AUTH_TOKEN"]


def get_model(
        model_path: str,
        model_load_params: dict,
        adapter_weights_path: Union[str, None] = None
        ) -> pipeline:
    """
    Get model pipeline for inference

    Args:
        model_path (str): LLM model path
        model_load_params (dict): model loading params
        adapter_weights_path (str, optional): If give, load adaptor weights.
                                              Defaults to None.

    Returns:
        pipeline: model pipeline
    """
    logging.info("Loading model and tokenizer")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        **model_load_params,
        use_auth_token=HF_AUTH_TOKEN
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        model_path,
        use_auth_token=HF_AUTH_TOKEN
    )

    if adapter_weights_path:
        logging.info("Loading adapter weights")
        model = PeftModel.from_pretrained(
            model,
            adapter_weights_path,
            torch_dtype=torch.float16
        )

    logging.info("Creating pipeline")
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return model_pipeline
