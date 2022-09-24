from src.preprocessing.token_classification import (
    preprocess,
    inference_preprocess
)

# preprocess("data", "train")

# preprocess("data", "val")

inference_preprocess("data", "test")