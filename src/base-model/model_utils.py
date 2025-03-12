from unsloth import FastVisionModel
from typing import Union, Literal

from config import get_config, PEFT_CONFIG


def initialize_model(mode: Union[Literal["inference"], Literal["finetuning"]] = "inference"):
    """Initialize and return the base model and tokenizer."""
    config = get_config(mode)
    model, tokenizer = FastVisionModel.from_pretrained(**config)
    return model, tokenizer

def setup_peft_model(model):
    """Apply PEFT configuration to the model."""
    return FastVisionModel.get_peft_model(model, **PEFT_CONFIG)