from os import path, makedirs
import pickle
import json
import torch
import shutil

from typing import Dict


def ensure_dir(path_: str) -> str:
    dir = path.dirname(path_)
    if not path.exists(dir):
        makedirs(dir)
    return path_

def save_obj_to_file(path_:str, obj:object):
    with open(ensure_dir(path_), "wb") as writer:
        pickle.dump(obj, writer)


def load_obj_from_file(path_: str) -> object:
    with open(path_, "rb") as reader:
        obj = pickle.load(reader)
    return obj

def save_data_to_json(path_:str, data: object):
    with open(ensure_dir(path_), "w", encoding="utf-8") as w:
        json.dump(data, w, indent=2, sort_keys=True, default=lambda o: o.__dict__)

def load_data_from_json(path_:str) -> object:
    with open(path_, "r", encoding="utf-8") as r:
        return json.load(r)

def save_checkpoint(path_:str, state: Dict, is_best: bool, filename="checkpoint.pth.tar"):
    torch.save(state, ensure_dir(path.join(path_, filename)))
    if is_best:
        shutil.copy(path.join(path_, filename), path.join(path_, "model_best.pth.tar"))
