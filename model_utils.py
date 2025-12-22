import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def get_llama(model_path, load_4bit=False, device="cpu"):
    """Load a LLaMA model from the given path.

    Args:
        model_path: Path to the model or HF model ID
        load_4bit: If True, load model in 4-bit quantization (requires bitsandbytes)
        device: Device to load model on (used only when load_4bit=False)

    Returns:
        Loaded model
    """

    def skip(*args, **kwargs):
        pass

    # Skip initialization for faster loading
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    # detect local path vs HF repo id; for local paths use local_files_only to avoid
    # HuggingFace hub repo-id validation errors when path starts with '/'
    is_local = os.path.exists(model_path)

    if load_4bit:
        # Load model in 4-bit to save memory (requires bitsandbytes)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=is_local,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": device},
            low_cpu_mem_usage=True,
            local_files_only=is_local,
        )

    model.seqlen = 2048
    return model


def dequantize_model(model):
    """Dequantize 4-bit modules to fp16 for weight access.

    This function walks through all modules and dequantizes any 4-bit
    quantized layers so their weights can be accessed as tensors.

    Args:
        model: Model to dequantize (modified in-place)
    """

    def _dequantize_module(m):
        # Try to dequantize bnb 4bit modules
        if hasattr(m, "dequantize"):
            try:
                m.dequantize()
                return
            except Exception:
                pass
        # Fallback: cast weight to fp16 if it's a parameter
        if hasattr(m, "weight") and isinstance(
            getattr(m, "weight", None), torch.nn.Parameter
        ):
            try:
                m.weight.data = m.weight.data.to(torch.float16)
            except Exception:
                pass

    model.apply(_dequantize_module)


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
