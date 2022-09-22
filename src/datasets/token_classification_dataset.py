import numpy as np
import torch
from typing import Sequence, Tuple, List

from src.utils.data import (
    OmniDataset,
    data_utils,
    EpochBatchIterator,
    ConcatDataset
)
import pyarrow as pa

def token_pad_bath(
        ids: Sequence[List[int]],
        sizes: Sequence[int],
        labels: Sequence[List[int]],
        pad_token_id: int,
        label_mask_token_id: int
) -> Tuple[np.array, np.array]:
    """
    Pad the instances to max length in the batch
    :param ids: sequences to pad
    :param sizes: size of each sequence
    :param pad: pad index
    :return:
    """

    max_len = max(sizes)
    batch_ids = np.array([np.array(seq + [pad_token_id] * (max_len - size)) for seq, size in zip(ids, sizes)])
    batch_labels = np.array([np.array(seq + [label_mask_token_id] * (max_len - size)) for seq, size in zip(labels, sizes)])
    return batch_ids, batch_labels

def token_collate(
        batch_samples: Tuple[Sequence[Sequence[int]], Sequence[int], Sequence[Sequence[int]]],
        pad_token_id: int,
        label_mask_token_id: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    collate function used to pad a batch of sampled examples
    :param batch_samples: examples to pad
    :param tensor_type: return tensor type
    :param pad: pad index
    :return:
    """
    input_ids, sizes, labels = zip(*batch_samples)
    input_ids, labels = token_pad_bath(
        ids=input_ids,
        sizes=sizes,
        labels=labels,
        pad_token_id=pad_token_id,
        label_mask_token_id=label_mask_token_id
    )
    return torch.LongTensor(input_ids), torch.FloatTensor(labels)


class TokenClassificationDataset(OmniDataset):
    def __init__(
        self,
        path: str,
        pad_token_id: int,
        label_mask_token_id: int,
        shuffle: bool,
    ):
        super(TokenClassificationDataset, self).__init__()
        self.shuffle = shuffle
        self.pad_token_id = pad_token_id
        self.label_mask_token_id = label_mask_token_id

        with pa.memory_map(path, 'r') as source:
            self.table = pa.ipc.open_file(source).read_all()

        # get sizes
        self.sizes = np.array(self.table.column("sizes").to_pylist())

        self.table = self.table.select([
            "input_ids",
            "labels",
        ])
        self.epoch = 0

    def __len__(self):
        return len(self.sizes)

    def __getitem__(
            self,
            item
    ) -> Tuple[List[int], int, int]:
        size = self.sizes[item]
        if isinstance(item, list) or isinstance(item, np.ndarray):
            batch = self.table.take(item).to_pydict()
            ids = batch.get("input_ids")
            label = batch.get("labels")
        else:
            batch = self.table.take([item]).to_pydict()
            ids = batch.get("input_ids")[0]
            label = batch.get("labels")[0]

        return (ids, size, label)

    def num_tokens(
            self,
            index: int
    ) ->int:
        return self.sizes[index]

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        return self.sizes[indices]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def collater(
        self,
        samples: Tuple[List[Sequence[int]], Sequence[int], List[Sequence[int]]]
    ):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        return token_collate(samples, self.pad_token_id, self.label_mask_token_id)

    def ordered_indices(self) -> np.array:
        """
        Get a list or example's indixes ordered randomly or by sizes
        """
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
            indices = indices[np.argsort(self.sizes[indices], kind='mergesort')]
        return indices

    
    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.epoch = epoch

    @property
    def supports_prefetch(self):
        return False


def get_token_classfication_dataset(
    paths: Sequence[str],
    max_tokens_per_batch: int,
    pad_token_id: int,
    seed: int = 0,
    label_mask_token_id: int = -100,
    max_sentence_length: int = 300,
    max_sentences_per_batch: int = 100,
    num_gpus: int = 1,
    max_iter_length: int = 0,
    num_workers: int = 0,
):
    if len(paths) == 1:
        dataset = TokenClassificationDataset(
            path=paths[0],
            pad_token_id=pad_token_id,
            label_mask_token_id=label_mask_token_id,
            # optimized batch completness
            shuffle=False,
        )
    else:
        dataset = [
            TokenClassificationDataset(
                path=p,
                pad_token_id=pad_token_id,
                label_mask_token_id=label_mask_token_id,
                shuffle=False,
            ) for p in paths
        ]
        dataset = ConcatDataset(dataset)


    with data_utils.numpy_seed(seed):
        indices = dataset.ordered_indices()

    # filter sentences too long
    indices, ingored = dataset.filter_indices_by_size(
        indices,
        min((max_sentence_length, max_tokens_per_batch))
    )

    # create mini-batch with given size constraint
    batch_sampler = dataset.batch_by_size(
        indices,
        max_tokens=max_tokens_per_batch,
        max_sentences=max_sentences_per_batch,
        required_batch_size_multiple=num_gpus
    )

    # return a reusable iterator
    epoch_iter = EpochBatchIterator(
        dataset=dataset,
        collate_fn=dataset.collater,
        batch_sampler=batch_sampler,
        seed=seed,
        num_workers=num_workers,
        buffer_size=200,
        max_iter_len=max_iter_length
    )

    return epoch_iter