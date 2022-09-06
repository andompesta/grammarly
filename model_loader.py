from src.utils import get_tokenizer
from src.models import get_model

if __name__ == "__main__":
    tokenizer = get_tokenizer("distilroberta-base")
    model = get_model("distilroberta-base", num_labels=1)
    model 


