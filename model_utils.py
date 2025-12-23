import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


def get_llama(
    model_path: str,
    device: str = "cpu",
):
    """Load a LLaMA model for merging.
    
    Always loads on CPU in FP32 for stable merging operations.
    For inference, load the merged model separately on GPU.

    Args:
        model_path: Path to the model or HF model ID
        device: Device to load model on (default: "cpu")

    Returns:
        Loaded model in FP32 on CPU
    """
    # Detect local path vs HF repo id
    is_local = os.path.exists(model_path)

    print(f"Loading model from {model_path} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=is_local,
    )
    
    model = model.to(device)
    model.seqlen = 2048
    return model


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find all layers of specified types within a module.
    Returns dictionary mapping layer names to layer modules.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res
