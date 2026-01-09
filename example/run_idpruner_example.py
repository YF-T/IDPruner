# -*- coding: utf-8 -*-
"""
================================================================================
|       IDPruner (MMR) Inference Example Script (`run_idpruner_example.py`)    |
================================================================================
File Description:
1. Demonstrates usage of IDPruner (MMR) using the specific definition key.
2. Dynamically maps the method key to its generated configuration.
3. Executes a full lifecycle: Setup -> Prune -> Inference -> Recover.
"""

import os
import sys
import torch
from PIL import Image
from transformers import AutoProcessor

# --- Dynamically add the project root directory to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import core components and configuration definitions ---
from pruning.pruning_modules import enable_pruning, recover_original_model
from pruning.pruning_cache import PruningCache
from pruning.configs.models import MODEL_CONFIGS
from pruning.configs.heuristic_methods import (
    HEURISTIC_METHOD_DEFINITIONS,
    generate_pruning_method_configs
)

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
# Target model and asset paths
MODEL_NAME = "Qwen2.5-VL-7B-Instruct"
IMAGE_PATH = "example/image.png"

# Pruning parameters
PRUNING_RATIO = 0.5
PRUNING_LAYER = "0"
# This key must match a key in HEURISTIC_METHOD_DEFINITIONS
METHOD_KEY = "idpruner_lambda0.5"

# Inference parameters
MAX_NEW_TOKENS = 30
QUESTION = "What is in this image?"
# ==============================================================================

def main():
    # 1. Validation
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"❌ Image not found at: {IMAGE_PATH}")
    
    if METHOD_KEY not in HEURISTIC_METHOD_DEFINITIONS:
        raise KeyError(f"❌ Method key '{METHOD_KEY}' is not defined in heuristic_methods.py")

    # 2. Map Method Key to Display Name Template
    # We need the template to find the specific config in the generated dictionary
    method_def = HEURISTIC_METHOD_DEFINITIONS[METHOD_KEY]
    template = method_def["display_name_template"]
    
    # Standard format logic used by the framework:
    # Note: For MMR methods, the template usually expects {ratio} and might have fixed lambda/k
    # We attempt to format it to get the 'Display Name' used as a key in all_configs
    try:
        # Most templates in heuristic_methods use {ratio}
        target_display_name = template.format(ratio=PRUNING_RATIO)
    except KeyError:
        # Fallback if the template doesn't use standard formatting (should not happen for MMR)
        target_display_name = template

    # 3. Dynamically generate pruning configurations
    print(f"⚙️  Generating pruning configs for {MODEL_NAME} (Ratio: {PRUNING_RATIO})...")
    all_generated_configs = generate_pruning_method_configs(
        MODEL_NAME, 
        PRUNING_LAYER, 
        PRUNING_RATIO
    )
    
    # 4. Extract the exact pruning configuration
    pruning_config = all_generated_configs.get(target_display_name)
            
    if pruning_config is None:
        raise ValueError(
            f"❌ Configuration for '{target_display_name}' was not generated.\n"
            f"Check if the ratio or model is supported by this method."
        )

    print(f"✅ Method Resolved: {METHOD_KEY} -> {target_display_name}")

    # 5. Load base model and processor
    print(f"\n📥 Loading model: {MODEL_NAME}...")
    model_meta = MODEL_CONFIGS[MODEL_NAME]
    model = model_meta["model_class"].from_pretrained(
        model_meta["path"],
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    ).eval()
    processor = AutoProcessor.from_pretrained(model_meta["path"], trust_remote_code=True)

    # 6. Prepare Multi-Modal Input
    raw_image = Image.open(IMAGE_PATH)
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image"}, 
                {"type": "text", "text": QUESTION}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[prompt], 
        images=[raw_image], 
        padding=True, 
        return_tensors="pt"
    ).to("cuda")

    # 7. Activate IDPruner Plugin
    print(f"\n🚀 Activating IDPruner plugin using {METHOD_KEY}...")
    # This logic replaces model layers with prunable versions and injects context
    pruned_model = enable_pruning(model, pruning_config)
    
    # 8. Execute Inference with PruningCache
    # PruningCache is required to handle KV synchronization during the decoding loop
    cache = PruningCache(config=pruned_model.config)
    
    print("🔮 Model is generating response...")
    with torch.no_grad():
        output_ids = pruned_model.generate(
            **inputs,
            use_cache=True,
            past_key_values=cache,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    # 9. Result Decoding
    prompt_length = inputs.input_ids.shape[1]
    generated_text = processor.batch_decode(
        output_ids[:, prompt_length:], 
        skip_special_tokens=True
    )[0]
    
    print(f"\n" + "="*50)
    print(f"IMAGE:    {IMAGE_PATH}")
    print(f"KEY:      {METHOD_KEY}")
    print(f"PROMPT:   {QUESTION}")
    print(f"RESPONSE: {generated_text}")
    print("="*50)

    # 10. Clean up and restore model
    recover_original_model(model)
    print("\n✅ Lifecycle complete: Model weights restored and plugin detached.")

if __name__ == "__main__":
    main()