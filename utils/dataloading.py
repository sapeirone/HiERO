import torch
import random
import numpy as np

import os

from typing import Tuple, Union, Dict, List

from torch_geometric.loader import DataLoader
    

class InfiniteLoader:
    def __init__(self, loaders: Union[List[DataLoader], Dict[str, DataLoader]]):
        if isinstance(loaders, list):
            loaders = {str(i): loader for i, loader in enumerate(loaders)}
            self.return_as_tuple = True
        else:
            self.return_as_tuple = False
            
        self.loaders = loaders
        # create an iterator for each loader
        self.iterators = {name: iter(loader) for name, loader in self.loaders.items()}

    def __iter__(self):
        return self
    
    def __len__(self):
        return max(len(loader) for loader in self.loaders.values())

    def __next__(self):
        data = dict()

        # iterate over all iterators
        for name, iterator in self.iterators.items():

            try:
                # try loading from this iterator
                data[name] = next(iterator)  # type: ignore
            except StopIteration:
                # some iterators still need to complete -> reset this iterator and keep looping
                self.iterators[name] = iter(self.loaders[name])
                data[name] = next(self.iterators[name])  # type: ignore

        if self.return_as_tuple:
            return tuple(data.values())
        
        return data


def worker_init_reset_seed(_):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_dataloader(dataset, batch_size, is_training, num_workers, drop_last, rng_generator=None):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        generator=rng_generator,
        follow_batch=['labels', 'segments', 'narration_timestamps', 'extra_narrations']
    )
