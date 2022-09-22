from transformers import PreTrainedTokenizerFast
from src.utils import get_tokenizer
from typing import *
import pyarrow as pa
import pyarrow.dataset as ds
import os

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
    # tokenize
    tokenizer = get_tokenizer("distilroberta-base")
    input_ids = []
    sizes = []
    labels = []
    tags = []
    idxs = []
    shards = []
    
    for example in group:
        # shard used for filtering
        shards.append(example["shard"])
        # from token to token_ids
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



def main(
    base_path: str,
    group_name: str,
):
    if group_name == "train":
        shards = [
            (f"{group_name}srcac", f"{group_name}lblac"),
            (f"{group_name}srcaa", f"{group_name}lblaa"),
            (f"{group_name}srcab", f"{group_name}lblab"),
        ]
    elif group_name == "val":
        shards = (f"{group_name}.src", f"{group_name}.lbl")


    for i, (src_file_name, lbl_file_name) in enumerate(shards):
        dataset = []
        src_path = os.path.join(
            base_path,
            src_file_name,
        )
        lbl_path = os.path.join(
            base_path,
            lbl_file_name,
        )
        with open(src_path, "r") as src_file, open(lbl_path, "r") as lbl_file:
            for idx, (src_line, lbl_line) in enumerate(zip(src_file, lbl_file)):
                # read each line
                if src_line == "\n":
                    # skip empty lines
                    assert src_line == lbl_line
                    continue
                
                src_tokens = src_line.strip().split()
                tags = [int(t) for t in lbl_line.strip().split()]
                assert len(src_tokens) == len(tags)

                # parse tokens of each example
                example = dict(
                    tokens=src_tokens,
                    tags=tags,
                    idx=idx,
                    shard=i,
                )
                if sum(tags) > 0:
                    dataset.append(example)

                if idx % 10000 == 0:
                    # logging
                    print(idx)
        
        dataset = encode(dataset)
        # convert to pyarrow table
        dataset = pa.table(dataset)


        batches = dataset.to_batches()
        with pa.OSFile("./{}/{}/shard_{}.arrow".format(base_path, group_name, i), 'wb') as sink:
            # Get the first batch to read the schema
            with pa.ipc.new_file(sink, schema=batches[0].schema) as writer:
                for batch in batches:
                    writer.write(batch)
