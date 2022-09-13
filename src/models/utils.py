import os
from transformers import PreTrainedModel, RobertaForTokenClassification

def get_model(
    mode_name: str,
    **kwargs
) -> PreTrainedModel:
    if mode_name == "distilroberta-base":
        model = RobertaForTokenClassification.from_pretrained(
            "distilroberta-base",
            cache_dir=os.path.join(
                "./models/pretrained",
                mode_name
            ),
            **kwargs
        )
    else:
        raise NotImplementedError("model {} not supported yet".format(mode_name))
    return model