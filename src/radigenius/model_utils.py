from unsloth import FastVisionModel
from typing import Union, Literal

from config import get_config


def initialize_model(mode: Union[Literal["inference"], Literal["finetuning"]] = "inference"):
    """Initialize and return the base model and tokenizer."""
    config = get_config(mode)
    model, tokenizer = FastVisionModel.from_pretrained(**config)
    return model, tokenizer
