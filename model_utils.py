import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def save_model_with_tokenizer(model, save_path: str, tokenizer_source: str):
    """
    Save a merged model along with its tokenizer.

    Args:
        model: The model to save
        save_path: Directory to save the model and tokenizer
        tokenizer_source: Path or model ID to load the tokenizer from
    """
    print(f"Saving merged model to {save_path}...")
    model.save_pretrained(save_path)

    print(f"Saving tokenizer from {tokenizer_source}...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=os.path.exists(tokenizer_source),
    )
    tokenizer.save_pretrained(save_path)

    print(f"✓ Model and tokenizer saved to {save_path}")


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
