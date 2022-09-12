from typing import Optional
import torch
import numpy as np
import argparse
import os
from datetime import datetime
import wandb

from src.models import get_model
from src.datasets import get_token_classfication_dataset
from src.optim import (
    get_optimizer,
    get_group_params,
    get_linear_scheduler_with_warmup,
    unfreeze_layer_params,
)
from src.utils import save_checkpoint
from src.tasks import TokenClassificationTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--training_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, required=True)
    parser.add_argument("--ckp_path", type=str, required=True)
    parser.add_argument("--shard_id", type=int, default=None)
    
    parser.add_argument("--task_name", default="token_classification")
    parser.add_argument("--model_version", default="distilroberta-base")
    parser.add_argument("--db_name", default="grammarly")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_gpus", default=1)

    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--unfreeze_layer", default=3, type=int)
    parser.add_argument("--batches_per_epoch", default=100, type=int)
    parser.add_argument("--max_sentences_per_batch", default=200, type=int)
    parser.add_argument("--max_tokens_per_batch", default=5000, type=int)
    parser.add_argument("--max_sentence_length", default=256, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--optim_method", default="adam")
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--eval_every", default=5, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    return parser.parse_args()


def compute_warmup_steps(
    args: argparse.Namespace, warmup_persentage: float = 1.5
) -> argparse.Namespace:
    args.steps_per_epoch = int(
        args.batches_per_epoch / args.gradient_accumulation_steps
    )
    args.num_warmup_steps = args.steps_per_epoch * warmup_persentage
    args.num_training_steps = int(args.steps_per_epoch * args.epochs)
    return args


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    args = parse_args()
    device = torch.device(args.device)
    
    model = get_model(args.model_version, num_labels=1)
    
    train_epoch_gen = get_token_classfication_dataset(
        args.training_path,
        max_tokens_per_batch=args.max_tokens_per_batch,
        pad_token_id=model.config.pad_token_id,
        max_iter_length=args.batches_per_epoch,
        max_sentence_length=args.max_sentence_length,
        max_sentences_per_batch=args.max_sentences_per_batch,
        num_gpus=args.n_gpus,
        shard_id=args.shard_id
    )

    eval_epoch_gen = get_token_classfication_dataset(
        args.valid_path,
        max_tokens_per_batch=args.max_tokens_per_batch,
        pad_token_id=model.config.pad_token_id,
        max_sentence_length=args.max_sentence_length,
        max_sentences_per_batch=args.max_sentences_per_batch,
        num_gpus=args.n_gpus,
    )

    if args.batches_per_epoch != len(train_epoch_gen):
        args.batches_per_epoch = len(train_epoch_gen)
    
    args = compute_warmup_steps(args)
    model_name = model.config._name_or_path
    db_name = args.db_name
    
    exp_name = f'{args.run_name}-{args.task_name}-{db_name}-{model_name}-{datetime.now().strftime("%d-%m-%y_%H-%M-%S")}'
    print(f"RUNNING EXPERIMENT -> {exp_name}")

    with wandb.init(
        project="grammarly",
        name=exp_name,
        job_type=args.task_name,
        notes=args.notes,
        config=vars(args),
    ) as run:


        # setup optimizers
        named_params = list(model.named_parameters())
        group_params = get_group_params(
            named_params,
            args.weight_decay,
            no_decay=["bias", "layer_norm.weight", "layer_norm.bias"],
        )
        unfreeze_layer_params(named_params, layer=args.unfreeze_layer)
        optim = get_optimizer(
            method=args.optim_method,
            params=group_params,
            lr=args.lr,
        )
        scheduler = get_linear_scheduler_with_warmup(
            optim, 
            args.num_warmup_steps,
            args.num_training_steps,
        )

        model = model.to(device)
        

        if torch.cuda.device_count() > 1 and args.n_gpus > 1:
            model = torch.nn.DataParallel(model, device_ids=[1, 0])
            args.max_sentences_per_batch *= args.max_sentences_per_batch
            args.max_tokens_per_batch *= (args.n_gpus // 2)
    
        best_f1 = 0.0
        train_loss = []
        train_acc = []

        eval_loss = []
        eval_prec = []
        eval_recal = []
        eval_f1 = []
        eval_acc = []

        task = TokenClassificationTask(
            name=exp_name,
            args=args,
            pad_token_id=model.config.pad_token_id,
        )

        for epoch in range(1, args.epochs + 1):
            train_iter_dl = train_epoch_gen.next_epoch_itr(shuffle=True)
            loss, acc = task.train(
                model=model,
                optimizer=optim,
                scheduler=scheduler,
                dataloader=train_iter_dl,
                device=device,
            )

            train_loss.append(loss)
            train_acc.append(acc)
            print(f"epoch:{epoch}\tacc:{acc} \t loss:{loss}")
            run.log(dict(train_loss=loss, train_accuracy=acc), step=epoch - 1)

            if epoch % args.eval_every == 0 or epoch == 1:
                is_best = False
                eval_iter_dl = eval_epoch_gen.next_epoch_itr(shuffle=True)
                loss, (acc, prec, rec, f_score), _ = task.eval(
                    model=model,
                    dataloader=eval_iter_dl,
                    device=device,
                )

                eval_loss.append(loss)
                eval_acc.append(acc)
                eval_prec.append(prec)
                eval_recal.append(rec)
                eval_f1.append(f_score)
                print(
                    f"--------->eval\tacc:{acc}\tloss{loss}\tprec:{prec}\trec:{rec}\tf1:{f_score}"
                )
                run.log(
                    dict(
                        eval_loss=loss,
                        eval_accuracy=acc,
                        eval_precision=prec,
                        eval_recal=rec,
                        eval_f_1=f_score,
                    ),
                    step=epoch - 1,
                )

                if f_score > best_f1:
                    best_f1 = f_score
                    is_best = True

                if isinstance(model, torch.nn.DataParallel):
                    state_dict = dict(
                        [(n, p.to("cpu")) for n, p in model.module.state_dict().items()]
                    )
                else:
                    state_dict = dict(
                        [(n, p.to("cpu")) for n, p in model.state_dict().items()]
                    )

                save_checkpoint(
                    path_=os.path.join(
                        args.ckp_path, args.task_name, db_name, model_name
                    ),
                    state=state_dict,
                    is_best=is_best,
                    filename=f"ckp_{epoch}.pth.tar",
                )
