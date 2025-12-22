# Model Merging - Clean Architecture

This codebase implements TIES and DARE model merging with clean separation of concerns.

## File Structure

```
├── model_prep.py      # Preprocessing: vocab align, quantization removal, dtype conversion
├── ties_utils.py      # Pure TIES algorithm implementation
├── dare_utils.py      # Pure DARE algorithm implementation
├── ties_llama.py      # TIES merging for LLaMA models
├── dare_llama.py      # DARE merging for LLaMA models
└── model_utils.py     # Model loading and utility functions
```

## Architecture Principles

### 1. Preprocessing (`model_prep.py`)

- **Handles**: Vocab size alignment, quantization removal, dtype conversion
- **Input**: Raw models (may have quantization, different vocabs, mixed dtypes)
- **Output**: Clean, aligned FP16 models ready for merging

### 2. Merging (`ties_llama.py`, `dare_llama.py`)

- **Handles**: Pure parameter merging logic
- **Input**: Preprocessed models (aligned vocabs, same dtype, no quantization)
- **Output**: Merged model with same structure as base model

### 3. Utils (`ties_utils.py`, `dare_utils.py`)

- **Handles**: Core merging algorithms (task vectors, trimming, election)
- **Input/Output**: Pure tensor operations

## Usage

### Basic Merging (FP16 models)

```python
from ties_llama import ties_merge_llama

merged = ties_merge_llama(
    base_model_path="model/llama2-7b-fp16",
    ft_model_paths=["model/codellama-7b-fp16", "model/metamath-7b-fp16"],
    weights=[0.5, 0.5],
    densities=[1.0, 1.0],
    device="cuda",
    use_fp16=True  # Models are already FP16
)

merged.save_pretrained("merged_model")
```

### Advanced: Manual Preprocessing

If you need custom preprocessing:

```python
from model_utils import get_llama
from model_prep import prepare_models_for_merging
from ties_utils import TIES

# Load models
base = get_llama("base_model", load_4bit=False)
ft1 = get_llama("ft_model_1", load_4bit=False)
ft2 = get_llama("ft_model_2", load_4bit=False)

# Preprocess (handles all edge cases)
base, [ft1, ft2] = prepare_models_for_merging(base, [ft1, ft2])

# Now merge manually if needed
ties = TIES()
# ... merge layer by layer
```

## What Preprocessing Handles

- ✅ **Vocabulary size mismatches** → Aligns all models to minimum vocab
- ✅ **Quantized modules** → Replaces Linear4bit with nn.Linear
- ✅ **Mixed dtypes** → Converts all parameters to FP16 (or specified dtype)
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
