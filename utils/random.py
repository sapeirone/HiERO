"""Seed everything and ensure deterministic execution."""

import logging
import os
import random

import numpy as np
import torch
from torch.backends import cudnn

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> torch.Generator:
    """Seed everything and ensure deterministic execution (as much as possible).

    Parameters
    ----------
    seed : int
        Seed number for torch, numpy and python RNGs.

    Returns
    -------
    torch.Generator
        The random number generator for torch.
    """
    logger.debug("Seeding everything with seed %d.", seed)

    torch_rng = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # training: disable cudnn benchmark to ensure the reproducibility
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # this is needed for CUDA >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    return torch_rng
