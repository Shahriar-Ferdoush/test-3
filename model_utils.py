import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def get_llama(
    model_path,
    load_4bit: bool = False,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.float16,
    low_cpu_mem_usage: bool | None = None,
):
    """Load a LLaMA model from the given path.

    Args:
        model_path: Path to the model or HF model ID
        load_4bit: If True, load model in 4-bit quantization (requires bitsandbytes)
        device: Device to load model on (used only when load_4bit=False)
        torch_dtype: dtype to load weights into (Transformers arg name is torch_dtype)
        low_cpu_mem_usage: If None, choose a safe default:
            - CPU: False (avoids meta-tensor edge cases during preprocessing)
            - non-CPU: True

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

    device_str = str(device)
    is_cpu = device_str == "cpu" or device_str.startswith("cpu")

    if low_cpu_mem_usage is None:
        low_cpu_mem_usage = False if is_cpu else True

    if load_4bit:
        # Load model in 4-bit to save memory (requires bitsandbytes)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=low_cpu_mem_usage,
            local_files_only=is_local,
            torch_dtype=torch_dtype,
        )
    else:
        # On CPU: avoid device_map and (by default) avoid low_cpu_mem_usage to prevent meta leftovers
        if is_cpu:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=None,
                low_cpu_mem_usage=low_cpu_mem_usage,
                local_files_only=is_local,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map={"": device_str},
                low_cpu_mem_usage=low_cpu_mem_usage,
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


def replace_linear4bit_with_linear(model):
    """
    Replace all Linear4bit (bitsandbytes quantized) modules with standard nn.Linear.
    This ensures the model can be saved and loaded without bitsandbytes dependencies.

    Args:
        model: Model to convert (modified in-place)
    """
    import torch.nn as nn

    replacements = []

    # Find all modules that need replacement
    for name, module in model.named_modules():
        # Check if it's a bitsandbytes Linear4bit module
        if (
            module.__class__.__name__ in ["Linear4bit", "Linear8bitLt"]
            or "bitsandbytes" in module.__class__.__module__
        ):
            replacements.append((name, module))

    # Replace each quantized module with standard Linear
    for name, old_module in replacements:
        # Get module attributes
        in_features = (
            old_module.in_features
            if hasattr(old_module, "in_features")
            else old_module.weight.shape[1]
        )
        out_features = (
            old_module.out_features
            if hasattr(old_module, "out_features")
            else old_module.weight.shape[0]
        )
        has_bias = old_module.bias is not None

        # Create new standard Linear module
        new_module = nn.Linear(in_features, out_features, bias=has_bias)

        # Copy weights (dequantize if needed)
        if hasattr(old_module, "weight"):
            with torch.no_grad():
                weight_data = old_module.weight.data
                # Ensure weight is FP16 tensor
                if not isinstance(
                    weight_data, torch.Tensor
                ) or weight_data.dtype not in [torch.float16, torch.float32]:
                    try:
                        weight_data = weight_data.to(torch.float16)
                    except:
                        # If conversion fails, try dequantization
                        if hasattr(old_module, "dequantize"):
                            old_module.dequantize()
                        weight_data = old_module.weight.data.to(torch.float16)

                new_module.weight.data = weight_data.clone().to(torch.float16)

        # Copy bias if exists
        if has_bias and old_module.bias is not None:
            with torch.no_grad():
                new_module.bias.data = old_module.bias.data.clone().to(torch.float16)

        # Replace the module in the model
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]

        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model

        setattr(parent, child_name, new_module)

    if replacements:
        print(
            f"✓ Replaced {len(replacements)} quantized modules with standard nn.Linear"
        )

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
