import os
from transformers import PreTrainedTokenizer, AutoTokenizer


def get_tokenizer(
    model_name: str,
    cache_dir: str = "./models/tokenizers"
) -> PreTrainedTokenizer:
    if model_name == "distilroberta-base":
        tokenizer = AutoTokenizer.from_pretrained(
            'roberta-base',
            add_prefix_space=True,
            cache_dir=os.path.join(
                cache_dir,
                model_name
            )
        )
    else:
        raise NotImplementedError("model {} not supprted".format(model_name))
    
    return tokenizer