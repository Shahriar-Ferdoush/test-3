# Model Merging - Clean Architecture

This codebase implements TIES and DARE model merging with clean separation of concerns.

**Target:** 1B-3B parameter models  
**Merging:** CPU (FP32) - stable, no FP16 issues  
**Inference:** GPU (FP16) - fast

## File Structure

```
├── model_prep.py      # Preprocessing: vocab align
├── ties_utils.py      # Pure TIES algorithm implementation
├── dare_utils.py      # Pure DARE algorithm implementation
├── ties_llama.py      # TIES merging for LLaMA models
├── dare_llama.py      # DARE merging for LLaMA models
└── model_utils.py     # Model loading utilities
```

## Architecture Principles

### 1. Preprocessing (`model_prep.py`)

- **Handles**: Vocab size alignment
- **Input**: Raw models (may have different vocabs)
- **Output**: Clean, aligned models ready for merging

### 2. Merging (`ties_llama.py`, `dare_llama.py`)

- **Handles**: Pure parameter merging logic
- **Input**: Models loaded on CPU in FP32
- **Output**: Merged model (saved to disk)

### 3. Inference (separate step)

- **Load merged model on GPU in FP16**
- **Run fast inference**

### 3. Utils (`ties_utils.py`, `dare_utils.py`)

- **Handles**: Core merging algorithms (task vectors, trimming, election)
- **Input/Output**: Pure tensor operations

## Usage

### Basic Merging (1B-3B Models)

```python
from ties_llama import ties_merge_llama

# Step 1: Merge on CPU (uses FP32, stable)
merged = ties_merge_llama(
    base_model_path="TinyLlama/TinyLlama-1.1B",
    ft_model_paths=["model1-1b-code", "model2-1b-math"],
    weights=[0.5, 0.5],
    densities=[1.0, 1.0],
    device="cpu"
)

# Step 2: Save merged model
merged.save_pretrained("merged_model")

# Step 3: Load on GPU for inference
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "merged_model",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### Works with Kaggle Paths

```python
# Merge on Kaggle CPU
merged = ties_merge_llama(
    base_model_path="/kaggle/input/llama-3.2/transformers/1b-instruct/1",
    ft_model_paths=[
        "/kaggle/input/model-1/finetuned",
        "/kaggle/input/model-2/finetuned"
    ],
    weights=[0.5, 0.5],
    densities=[1.0, 1.0],
    device="cpu"
)

# Save to working directory
merged.save_pretrained("/kaggle/working/merged_model")

# Load on GPU for inference
model = AutoModelForCausalLM.from_pretrained(
    "/kaggle/working/merged_model",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### Advanced: Manual Preprocessing

```python
from model_utils import get_llama
from model_prep import prepare_models_for_merging
from ties_utils import TIES

# Load models on CPU
base = get_llama("base_model", device="cpu")
ft1 = get_llama("ft_model_1", device="cpu")
ft2 = get_llama("ft_model_2", device="cpu")

# Preprocess (handles vocab alignment)
base, [ft1, ft2] = prepare_models_for_merging(base, [ft1, ft2])

# Merge layer by layer
ties = TIES()
# ... merge manually
```

## What Preprocessing Handles

- ✅ **Vocabulary size mismatches** → Aligns all models to minimum vocab
- ✅ **Validation** → Ensures models are compatible

## Workflow

```
1. Load models on CPU (FP32)
   ↓
2. Merge parameters (CPU arithmetic)
   ↓
3. Save merged model to disk
   ↓
4. Load on GPU (FP16) for fast inference
```

- ✅ **Input embeddings** → Aligns embed_tokens with lm_head

## Design Goals

1. **Single Responsibility**: Each file has one clear purpose
2. **Pure Functions**: Merging logic is pure tensor math
3. **Predictable Output**: Merged model = same structure as base model
4. **Easy Debugging**: Each stage can be inspected independently

## Testing

```python
# Verify models are preprocessed correctly
from model_prep import prepare_models_for_merging

base, ft_models = prepare_models_for_merging(base, ft_models)

# Check vocab sizes match
assert base.config.vocab_size == ft_models[0].config.vocab_size
assert base.lm_head.weight.shape[0] == ft_models[0].lm_head.weight.shape[0]

# Check no quantization
for name, module in base.named_modules():
    assert "bitsandbytes" not in module.__class__.__module__
```
