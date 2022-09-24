import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse

from src.models import get_model
from src.datasets import TokenClassificationDataset
from src.tasks import TokenClassificationTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, required=True, help="file containing the test dataset")
    parser.add_argument("--pred_path", type=str, required=True, help="path where to store the predicted labels")
    parser.add_argument("--ckp_path", type=str, required=True, help="path of the checkpoint to load")
    
    parser.add_argument("--model_version", default="distilroberta-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_gpus", default=1)
    parser.add_argument("--num_workers", default=4, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    args = parse_args()
    device = torch.device(args.device)
    
    # load model
    model = get_model(args.model_version, num_labels=1)
    model.load_state_dict(torch.load(args.ckp_path, map_location="cpu"))
    model.eval()

    # load dataset
    test_dataset = TokenClassificationDataset(
        path=args.test_path,
        pad_token_id=model.config.pad_token_id,
        label_mask_token_id=-100,
        shuffle=False,
    )

    dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        collate_fn=test_dataset.collater,
    )
    model = model.to(device)

    
    preds = TokenClassificationTask.inference(
        model=model,
        dataloader=dataloader,
        device=device,
        pad_token_id=model.config.pad_token_id,
        ignore_index=-100
    )

    with open(args.pred_path, "w") as writer:
        for p in preds:
            writer.write(" ".join(map(str, p)) + "\n")

