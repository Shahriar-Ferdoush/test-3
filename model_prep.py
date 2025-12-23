"""
Model Preprocessing Utilities

Handles all model preparation tasks before merging:
- Vocabulary size alignment
- Weight type conversion (FP16)

All preprocessing happens here, so merge functions can be pure.
"""

from typing import List, Tuple

import torch
from transformers import PreTrainedModel


def align_vocab_sizes(
    base_model: PreTrainedModel, ft_models: List[PreTrainedModel]
) -> Tuple[PreTrainedModel, List[PreTrainedModel]]:
    """
    Align vocabulary sizes across all models by trimming to minimum common size.

    Args:
        base_model: Base model
        ft_models: List of fine-tuned models

    Returns:
        Tuple of (base_model, ft_models) with aligned vocabularies
    """
    # Get vocab sizes
    base_vocab = base_model.lm_head.weight.shape[0]
    ft_vocabs = [m.lm_head.weight.shape[0] for m in ft_models]
    all_vocabs = [base_vocab] + ft_vocabs

    if len(set(all_vocabs)) == 1:
        # All same, no alignment needed
        return base_model, ft_models

    # Find minimum vocab size
    min_vocab = min(all_vocabs)
    print(f"⚠️  Vocab size mismatch: {all_vocabs} → Aligning to {min_vocab}")

    # Trim base model if needed
    if base_vocab > min_vocab:
        base_model.lm_head.weight.data = base_model.lm_head.weight.data[:min_vocab, :]
        if hasattr(base_model, "model") and hasattr(base_model.model, "embed_tokens"):
            base_model.model.embed_tokens.weight.data = (
                base_model.model.embed_tokens.weight.data[:min_vocab, :]
            )
        base_model.config.vocab_size = min_vocab

    # Trim fine-tuned models if needed
    for i, (ft_model, ft_vocab) in enumerate(zip(ft_models, ft_vocabs)):
        if ft_vocab > min_vocab:
            ft_model.lm_head.weight.data = ft_model.lm_head.weight.data[:min_vocab, :]
            if hasattr(ft_model, "model") and hasattr(ft_model.model, "embed_tokens"):
                ft_model.model.embed_tokens.weight.data = (
                    ft_model.model.embed_tokens.weight.data[:min_vocab, :]
                )
            ft_model.config.vocab_size = min_vocab

    print(f"✓ All models aligned to vocab_size={min_vocab}")
    return base_model, ft_models


def ensure_dtype(
    model: PreTrainedModel, dtype: torch.dtype = torch.float16
) -> PreTrainedModel:
    """
    Ensure all model parameters are in specified dtype.

    Args:
        model: Model to convert
        dtype: Target dtype (default: float16)

    Returns:
        Model with all parameters in target dtype
    """
    for param in model.parameters():
        if param.dtype != dtype:
            param.data = param.data.to(dtype)

    return model


def prepare_models_for_merging(
    base_model: PreTrainedModel,
    ft_models: List[PreTrainedModel],
    target_dtype: torch.dtype = None,
) -> Tuple[PreTrainedModel, List[PreTrainedModel]]:
    """
    Complete preprocessing pipeline: align vocabs and ensure dtype.

    Args:
        base_model: Base model
        ft_models: List of fine-tuned models
        target_dtype: Target dtype for all parameters (default: None = keep current dtype)

    Returns:
        Tuple of (prepared_base_model, prepared_ft_models)
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING MODELS FOR MERGING")
    print("=" * 60)

    # Step 1: Align vocabulary sizes
    print("\n[1/2] Aligning vocabulary sizes...")
    base_model, ft_models = align_vocab_sizes(base_model, ft_models)

    # Step 2: Ensure consistent dtype (if specified)
    if target_dtype is not None:
        print(f"\n[2/2] Converting to {target_dtype}...")
        base_model = ensure_dtype(base_model, target_dtype)
        ft_models = [ensure_dtype(m, target_dtype) for m in ft_models]
    else:
        print(f"\n[2/2] Keeping original dtype (no conversion)")

    print("\n" + "=" * 60)
    print("✓ PREPROCESSING COMPLETE - Models ready for merging")
    print("=" * 60 + "\n")

    return base_model, ft_models
