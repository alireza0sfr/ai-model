import torch
from typing import Union, Literal

# ======================================================================
# INFERENCE-OPTIMIZED CONFIGURATION
# ======================================================================
# Model settings optimized for inference

INFERENCE_CONFIG = {
    "model_name": "0llheaven/Llama-3.2-11B-Vision-Radiology-mini",
    
    # For highest quality inference on RTX 4090 (24GB VRAM):
    # - False: Uses full precision for maximum quality
    # - True: More memory efficient but slightly lower quality
    "load_in_4bit": False,  # Set to False for highest quality on RTX 4090
    
    # Best precision for medical imaging on RTX 4090
    "dtype": torch.bfloat16,
    
    # Disable for inference (not needed, improves speed)
    "use_gradient_checkpointing": False,
    
    # Can use longer sequences for inference (fits in 24GB)
    "max_seq_length": 4096,
    
    # Same cache directory
    "cache_dir": "/base-model",
}

# ======================================================================
# FINE-TUNING OPTIMIZED CONFIGURATION
# ======================================================================
# Model with settings optimized for fine-tuning

FINETUNING_CONFIG = {
    "model_name": "0llheaven/Llama-3.2-11B-Vision-Radiology-mini",
    
    # Must be True for fine-tuning to fit in 24GB VRAM
    # - Reduces model weights from ~22GB to ~5.5GB
    # - Leaves room for gradients and optimizer states
    "load_in_4bit": True,
    
    # bfloat16 provides better training stability than float16
    "dtype": torch.bfloat16,
    
    # Essential for memory-efficient training
    # - "unsloth": Optimized for Llama models
    # - Reduces memory usage by recomputing activations during backprop
    "use_gradient_checkpointing": "unsloth",
    
    # Reduced for training to fit in memory
    # - Each token requires additional memory during training
    # - 2048 is sufficient for most medical imaging tasks
    "max_seq_length": 2048,
    
    # Same cache directory
    "cache_dir": "/base-model",
}

# ======================================================================
# HELPER FUNCTION TO SELECT CONFIGURATION
# ======================================================================

def get_config(mode: Union[Literal["inference"], Literal["finetuning"]] = "inference"):
    """
    Returns the appropriate configuration based on the specified mode.
    
    Args:
        mode: Either "inference" or "finetuning"
    
    Returns:
        Dict containing the configuration
    """
    if mode.lower() == "finetuning":
        return FINETUNING_CONFIG
    else:
        return INFERENCE_CONFIG

# ======================================================================
# QUICK REFERENCE FOR RTX 4090 (24GB VRAM)
# ======================================================================
"""
MEMORY USAGE ESTIMATES:
-----------------------
| Configuration                   | VRAM Usage | Fits in 24GB? |
|---------------------------------|------------|---------------|
| Full precision inference        | ~22GB      | ✅ Yes        |
| 4-bit quantized inference       | ~6GB       | ✅ Yes        |
| Full precision fine-tuning      | ~66GB      | ❌ No         |
| 4-bit quantized fine-tuning     | ~18-22GB   | ✅ Yes        |

DTYPE RECOMMENDATIONS:
---------------------
| GPU Type                        | Recommended dtype    | Notes                    |
|---------------------------------|----------------------|--------------------------|
| RTX 4090, 3090, A100, H100      | torch.bfloat16      | Best stability & speed   |
| Other RTX 3000/4000 series      | torch.float16       | Good balance             |
| Older GPUs                      | torch.float16       | May need 4-bit always    |
"""