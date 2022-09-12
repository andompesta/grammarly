import imp
from .trainer import OmniTask
from .token_classification import TokenClassificationTask

__all__ = [
    "OmniTask",
    "TokenClassificationTask"
]