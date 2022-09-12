from transformers import PreTrainedTokenizerFast
from src.utils import get_tokenizer
from typing import *
import pyarrow as pa
import pyarrow.dataset as ds

def tokenize_and_align_labels(
    example: Sequence[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerFast,
    mask_idx: int = -100
):
    tokenized_inputs = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True,
        return_attention_mask=False,
        return_length=True
    )

    tags = example["tags"]
    word_ids = tokenized_inputs.word_ids()  # Map tokens to their respective word.

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:  
        # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append(mask_idx)

        elif word_idx != previous_word_idx:  
            # Only tags the first token of a given word.
            label_ids.append(tags[word_idx])

        else:
            label_ids.append(-100)

        previous_word_idx = word_idx

    tokenized_inputs["labels"] = label_ids
    return dict(
        **dict(tokenized_inputs),
        **dict(
            tags=tags,
            word_ids=word_ids,
            idx=example["idx"]
        )
    )


def encode(group):
    tokenizer = get_tokenizer("distilroberta-base")
    input_ids = []
    sizes = []
    labels = []
    tags = []
    idxs = []
    shards = []
    
    for example in group:
        shards.append(example["shard"])
        example = tokenize_and_align_labels(example, tokenizer)
        input_ids.append(example["input_ids"])
        sizes.append(example["length"][0])
        labels.append(example["labels"])
        tags.append(example["tags"])
        idxs.append(example["idx"])
    
    assert len(idxs) == len(input_ids) == len(sizes) == len(labels)

    return dict(
        input_ids=input_ids,
        sizes=sizes,
        labels=labels,
        tags=tags,
        idxs=idxs,
        shards=shards,
    )



def main(group_name: str):

    if group_name == "train":
        shards = [
            (f"{group_name}srcaa", f"{group_name}lblaa"),
            (f"{group_name}srcab", f"{group_name}lblab"),
            (f"{group_name}srcac", f"{group_name}lblac")
        ]
    elif group_name == "val":
        shards = (f"{group_name}.src", f"{group_name}.lbl")


    for i, (src_file_name, lbl_file_name) in enumerate(shards):
        dataset = []
        with open("data/" + src_file_name, "r") as src_file, open("data/" + lbl_file_name, "r") as lbl_file:
            for idx, (src_line, lbl_line) in enumerate(zip(src_file, lbl_file)):
                if src_line == "\n":
                    assert src_line == lbl_line
                    continue
                
                src_tokens = src_line.strip().split()
                tags = [int(t) for t in lbl_line.strip().split()]
                assert len(src_tokens) == len(tags)

                example = dict(
                    tokens=src_tokens,
                    tags=tags,
                    idx=idx,
                    shard=i,
                )
                dataset.append(example)

                if idx % 10000 == 0:
                    print(idx)
        
        dataset = encode(dataset)
        dataset = pa.table(dataset)
        ds.write_dataset(
            dataset,
            "./data/{}".format(group_name),
            format="parquet",
            partitioning=["shards", "sizes"],
            partitioning_flavor="hive",
            existing_data_behavior="delete_matching",
        )
