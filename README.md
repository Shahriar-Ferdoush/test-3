# TIES & DARE Model Merging for LLaMA

Clean implementation of TIES and DARE merging algorithms optimized for 1B-3B parameter models.

## Features

- ✅ **Simple API** - No complex parameters, just works
- ✅ **CPU Merging** - Merge on CPU in FP32 (stable, no FP16 issues)
- ✅ **GPU Inference** - Load merged model on GPU for fast inference
- ✅ **1B-3B Models** - Works with any size that fits in RAM
- ✅ **Automatic Preprocessing** - Handles vocab alignment
- ✅ **Clean Architecture** - Separation of concerns, easy to understand

## Quick Start

### Step 1: Merge on CPU

```python
from ties_llama import ties_merge_llama

# Merge on CPU (stable, uses FP32)
merged = ties_merge_llama(
    base_model_path="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    ft_model_paths=[
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "TinyLlama/TinyLlama-1.1B-python-v0.1"
    ],
    weights=[0.5, 0.5],
    densities=[1.0, 1.0],
    device="cpu"  # Always use CPU for merging
)

# Save merged model
merged.save_pretrained("merged_model")
print("✓ Merged model saved!")
```

### Step 2: Inference on GPU

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load merged model on GPU for fast inference
model = AutoModelForCausalLM.from_pretrained(
    "merged_model",
    torch_dtype=torch.float16,  # FP16 for GPU inference
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("merged_model")

# Run inference
prompt = "Write a Python function:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### DARE Merging

```python
from dare_llama import dare_merge_llama

# Merge on CPU
merged = dare_merge_llama(
    base_model_path="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    ft_model_paths=[
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "TinyLlama/TinyLlama-1.1B-python-v0.1"
    ],
    weights=[0.5, 0.5],
    densities=[0.9, 0.9],
    device="cpu"
)

merged.save_pretrained("merged_model")
```

### Works with Kaggle Paths

```python
# Merge on Kaggle CPU (uses system RAM, not GPU memory)
merged = ties_merge_llama(
    base_model_path="/kaggle/input/llama-3.2/transformers/1b-instruct/1",
    ft_model_paths=[
        "/kaggle/input/my-finetuned-model-1",
        "/kaggle/input/my-finetuned-model-2"
    ],
    weights=[0.5, 0.5],
    densities=[1.0, 1.0],
    device="cpu"  # Use CPU, not GPU
)

# Save to working directory
merged.save_pretrained("/kaggle/working/merged_model")

# Then load on GPU for inference
model = AutoModelForCausalLM.from_pretrained(
    "/kaggle/working/merged_model",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

## Recommended Models (1B-3B Range)

### TinyLlama (1.1B)

- Base: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- Chat: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Python: `TinyLlama/TinyLlama-1.1B-python-v0.1`

### Phi-2 (2.7B)

- Base: `microsoft/phi-2`
- Dolphin: `cognitivecomputations/dolphin-2_6-phi-2`

### StableLM-2 (1.6B)

- Base: `stabilityai/stablelm-2-1_6b`
- Zephyr: `stabilityai/stablelm-2-zephyr-1_6b`

## Memory Requirements

**Merging (CPU, FP32):**

| Model Size | Per Model | 3 Models Total | Kaggle RAM (30GB) |
| ---------- | --------- | -------------- | ----------------- |
| 1B         | ~4GB      | ~12GB          | ✅ Fits           |
| 2B         | ~8GB      | ~24GB          | ✅ Fits           |
| 3B         | ~12GB     | ~36GB          | ❌ Too large      |

**Inference (GPU, FP16):**

| Model Size | GPU Memory | Kaggle GPU (16GB) |
| ---------- | ---------- | ----------------- |
| 1B         | ~2GB       | ✅ Fast           |
| 2B         | ~4GB       | ✅ Fast           |
| 3B         | ~6GB       | ✅ Fast           |

**Recommended:** Merge 1B-2B models on Kaggle CPU, then run inference on GPU.

## Installation

```bash
pip install torch transformers accelerate
```

## Project Structure

```
├── model_utils.py     # Model loading utilities
├── model_prep.py      # Preprocessing (vocab alignment, dtype)
├── ties_utils.py      # TIES algorithm implementation
├── dare_utils.py      # DARE algorithm implementation
├── ties_llama.py      # TIES merging for LLaMA models
├── dare_llama.py      # DARE merging for LLaMA models
└── merge.py           # Unified merge interface
```

## How It Works

### TIES (TrIm, Elect Sign & Merge)

1. **Trim**: Remove small updates (based on density)
2. **Elect**: Resolve sign conflicts by voting
3. **Merge**: Average aligned updates

### DARE (Drop And REscale)

1. **Drop**: Randomly drop updates (based on density)
2. **Rescale**: Scale remaining updates to compensate
3. **Merge**: Combine rescaled updates

## Examples

See the notebooks in the repository:

- `ties-dare-merging.ipynb` - Complete example with TinyLlama models
- `fork-of-preprocess-llama.ipynb` - Preprocessing examples

## References

- **TIES-Merging**: [Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)
- **DARE**: [Language Models are Super Mario](https://arxiv.org/abs/2311.03099)

## License

MIT
