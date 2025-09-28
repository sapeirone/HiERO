"""Utilities for building optimizers and lr schedulers"""

import logging
from typing import Any, List

import hydra

from torch import nn
from torch import optim
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


def build_optimizer(models: List[nn.Module], cfg: Any) -> optim.Optimizer:
    """Build optimizer for a list of nn.Module modules.

    Parameters
    ----------
    models : List[torch.nn.Module]
        The list of models whose parameters should be part of the optimizer
    cfg : Any
        The configuration of the optimizer

    Returns
    -------
    Tuple[torch.optim.Optimizer, float]
        The instantiated optimizer and the number of trainable parameters (in millions)
    """

    # Collect all parameters that require gradients and set up the optimizer
    params = [param for m in models for param in m.parameters() if param.requires_grad]
    optimizer = hydra.utils.instantiate(
        cfg,
        [
            {"params": [p for p in params if len(p.shape) == 1], "lr": cfg.lr, "weight_decay": 0},
            {"params": [p for p in params if len(p.shape) > 1], "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        ],
    )

    nparams = sum(p.numel() for pg in optimizer.param_groups for p in pg["params"]) / 1e6
    logger.info("Model has %.4fM trainable parameters.", nparams)

    return optimizer


def build_lr_scheduler(
    lr_warmup: bool, num_epochs: int, lr_warmup_epochs: int, lr_min: float, dl_train: DataLoader, optimizer: optim.Optimizer
) -> tuple[optim.lr_scheduler.LRScheduler, optim.lr_scheduler.LRScheduler]:
    if lr_warmup:
        assert lr_warmup_epochs < num_epochs, "Warmup steps should be less than the total number of epochs."

        start_alpha, end_alpha = 0.001, 1
        logger.debug("Using lr warmup for %d epochs from %f to %f.", lr_warmup_epochs, start_alpha, end_alpha)

        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_alpha, end_alpha, lr_warmup_epochs * len(dl_train))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, (num_epochs - lr_warmup_epochs) * len(dl_train), lr_min)  # type: ignore

        return warmup_scheduler, scheduler

    return None, optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(dl_train), lr_min)  # type: ignore
