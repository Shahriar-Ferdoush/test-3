import argparse
from typing import List

import torch

from dare_utils import DARE
from model_prep import prepare_models_for_merging
from model_utils import dequantize_model, find_layers, get_llama


def dare_merge_llama(
    base_model_path: str,
    ft_model_paths: List[str],
    weights: List[float],
    densities: List[float],
    device: str = "cuda",
    use_fp16: bool = False,
):
    """
    Merge multiple fine-tuned LLaMA models using DARE method.

    Pure merging logic - assumes models are preprocessed and aligned.
    Use model_prep.prepare_models_for_merging() first if models have:
    - Different vocabulary sizes
    - Quantization
    - Mismatched dtypes

    Args:
        base_model_path: Path to the base model
        ft_model_paths: List of paths to fine-tuned models
        weights: List of weights for each fine-tuned model
        densities: List of densities for drop and rescale
        device: Device to use for computation
        use_fp16: If True, load models directly in FP16 (for preprocessed models).
                  If False, load in 4-bit and dequantize (default behavior).

    Returns:
        Merged LLaMA model
    """
    # === Load base model ===
    if use_fp16:
        print("Loading base model in FP16 (preprocessed)...")
        base_model = get_llama(base_model_path, load_4bit=False, device=device)
    else:
        print("Loading base model in 4-bit...")
        base_model = get_llama(base_model_path, load_4bit=True, device=device)
        print("Dequantizing base model...")
        dequantize_model(base_model)
    base_model.eval()

    # === Load all fine-tuned models ===
    print(f"Loading {len(ft_model_paths)} fine-tuned models...")
    ft_models = []
    for ft_path in ft_model_paths:
        if use_fp16:
            print(f"  → Loading {ft_path} in FP16 (preprocessed)...")
            model = get_llama(ft_path, load_4bit=False, device=device)
        else:
            print(f"  → Loading {ft_path} in 4-bit...")
            model = get_llama(ft_path, load_4bit=True, device=device)
            print(f"  → Dequantizing {ft_path}...")
            dequantize_model(model)
        model.eval()
        ft_models.append(model)

    # === Preprocessing: Align models ===
    print("\nPreprocessing models...")
    base_model, ft_models = prepare_models_for_merging(base_model, ft_models)

    # Initialize DARE merger
    dare = DARE()
    dev = torch.device(device)

    print("Starting layer-by-layer DARE merging...")

    # === Merge transformer layers sequentially ===
    layers = base_model.model.layers
    for i in range(len(layers)):
        print(f"Merging layer {i+1}/{len(layers)}...")

        # Find all Linear layers in this transformer layer
        subset = find_layers(layers[i])

        # Merge each Linear layer found (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
        for name in subset:
            print(f"  {name}")

            # Get base model layer weights
            base_weights = subset[name].weight.data

            # Get corresponding weights from all fine-tuned models
            ft_weights = []
            for ft_model in ft_models:
                # Navigate to the same layer in fine-tuned model
                ft_layer = ft_model.model.layers[i]
                ft_subset = find_layers(ft_layer)
                ft_weights.append(ft_subset[name].weight.data)

            # Apply DARE merging
            merged_weights = dare.merge(
                weights=weights,
                base_model_parameters=base_weights,
                ft_models_parameters=ft_weights,
                densities=densities,
                device=dev,
            )

            # Write merged weights back, ensuring contiguous and on correct device
            subset[name].weight.data = merged_weights.contiguous().to(device)

    # === Merge LM head ===
    print("Merging LM head...")
    base_lm_head = base_model.lm_head.weight.data
    ft_lm_heads = [m.lm_head.weight.data for m in ft_models]

    merged_lm_head = dare.merge(
        weights=weights,
        base_model_parameters=base_lm_head,
        ft_models_parameters=ft_lm_heads,
        densities=densities,
        device=dev,
    )

    # Ensure merged tensor is contiguous and on correct device before assignment
    base_model.lm_head.weight.data = merged_lm_head.contiguous().to(device)

    print("✓ DARE merging completed!")
    return base_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Base model path (required)
    parser.add_argument("base_model", type=str, help="Path to the base LLaMA model")

    # Fine-tuned models to merge (required)
    parser.add_argument(
        "--ft_models",
        nargs="+",
        type=str,
        required=True,
        help="Paths to fine-tuned models",
    )

    # Merging weights for each model (required)
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        required=True,
        help="Weights for each fine-tuned model",
    )

    # Density values for DARE drop and rescale (required)
    parser.add_argument(
        "--densities",
        nargs="+",
        type=float,
        required=True,
        help="Densities for drop and rescale (0-1, where 1 = no dropping)",
    )

    # Computation device
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )

    # Output path for merged model
    parser.add_argument(
        "--save", type=str, default="", help="Path to save the merged model"
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # === Validate inputs ===
    # Ensure all lists have matching lengths
    if len(args.ft_models) != len(args.weights):
        raise ValueError("Number of fine-tuned models must match number of weights")
    if len(args.ft_models) != len(args.densities):
        raise ValueError("Number of fine-tuned models must match number of densities")

    # === Perform DARE merging ===
    # Process model layer-by-layer and merge parameters
    merged_model = dare_merge_llama(
        base_model_path=args.base_model,
        ft_model_paths=args.ft_models,
        weights=args.weights,
        densities=args.densities,
        device=args.device,
    )

    # === Save merged model ===
    # Write merged model to disk if output path is specified
    if args.save:
        print(f"Saving merged model to {args.save}...")
        merged_model.save_pretrained(args.save)
        print("Model saved!")
