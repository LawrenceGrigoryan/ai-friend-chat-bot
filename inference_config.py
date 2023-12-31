"""
Model inference params
"""
import torch

MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
ADAPTER_WEIGHTS_PATH = None
MODEL_LOAD_PARAMS = {
    "device_map": "auto",
    "load_in_8bit": True,
    "torch_dtype": torch.float16
}
MODEL_INFERENCE_PARAMS = {
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 1.0,
    "num_return_sequences": 1,
    "max_new_tokens": 200,
    "repetition_penalty": 1.15,
    "use_cache": True
}
