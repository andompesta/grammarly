from src.utils import get_tokenizer
from src.models import get_model
import jax

if __name__ == "__main__":
    tokenizer = get_tokenizer("distilroberta-base")
    model = get_model("distilroberta-base", num_labels=1)
    print(model.config)
    rng = jax.random.PRNGKey(42)

    params = model.init_weights(rng, model.input_shape)


