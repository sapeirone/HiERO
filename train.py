"""Main entrypoint for HiERO training and validation on EgoClip/EgoMCQ."""

import logging
import os
import warnings
from typing import Optional

import hydra
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_  # type: ignore
from torchmetrics.aggregation import MeanMetric
from tqdm.auto import tqdm

from models.hiero import HiERO
from models.tasks.hiero import HiEROTask

from utils import (InfiniteLoader, build_dataloader, build_lr_scheduler,
                   build_optimizer, compute_hash, seed_everything)
from validate import validate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


warnings.filterwarnings("ignore")


def train(
    start_step: int,
    model: HiERO,
    task: HiEROTask,
    optimizer: optim.Optimizer,
    warmup_scheduler: Optional[optim.lr_scheduler.LRScheduler],
    lr_warmup_steps: int,
    scheduler: Optional[optim.lr_scheduler.LRScheduler],
    dataloader: InfiniteLoader,
    grad_clip: float = -1.0,
    num_steps: int = 1000,
    device: torch.device = torch.device("cpu"),
):
    """Run a number of training steps on EgoClip.

    Parameters
    ----------
    start_step : int
        Initial training step
    model : HiERO
        temporal model (e.g. TRN or GraphUNet)
    task : HiEROTask
        main task
    optimizer : torch.optim.Optimizer
        optimizer for the parameters
    warmup_scheduler : Optional[torch.optim.lr_scheduler.LRScheduler]
        warmup scheduler
    lr_warmup_steps : int
        number of warmup steps
    scheduler : Optional[torch.optim.lr_scheduler.LRScheduler]
        scheduler (post-warmup)
    dataloader : Iterator[Data]
        infinite dataloader over the input data
    grad_clip : float, optional
        gradient clip value, by default -1.0
    num_steps : int, optional
        number of training steps to perform, by default 1000
    device : str, optional
        device, by default "cuda"
    """
    loss_meter = MeanMetric().to(device)
    vna_loss_meter = MeanMetric().to(device)
    ft_loss_meter = MeanMetric().to(device)

    # Put models in train mode
    model.train()
    task.train()

    pbar = tqdm(range(start_step, start_step + num_steps), desc="Training on EgoClip...", leave=False)
    for step in pbar:
        (data,) = next(dataloader)  # type: ignore

        # Processing using the temporal graph
        graphs = model(data.to(device=device))
        outputs = task(graphs, data)

        # Compute the intermediate losses for each layer in the UNet
        loss, (vna_loss, ft_loss) = task.compute_loss(outputs, graphs, data)

        loss_meter.update(loss)
        vna_loss_meter.update(vna_loss)
        ft_loss_meter.update(ft_loss)

        # Backpropagate the sum of all the tasks' losses
        optimizer.zero_grad()
        loss.mean().backward()

        # Clip gradients (eventually)
        if grad_clip > 0:
            clip_grad_norm_([p for pg in optimizer.param_groups for p in pg["params"]], grad_clip)
        optimizer.step()

        # Update the learning rate according to the warmup scheduler
        if warmup_scheduler is not None and step <= lr_warmup_steps:
            warmup_scheduler.step(step - 1)

        # Update the learning rate according to the scheduler
        if scheduler is not None and step > lr_warmup_steps:
            scheduler.step(step - 1 - lr_warmup_steps)

        pbar.set_description(f"Training step {step} (total loss = {loss.mean().item():.4f}, vna = {vna_loss.mean().item():.4f}, ft = {ft_loss.mean().item():.4f}).")

    logger.info("Training loss = %.4f.", loss_meter.compute())
    logger.info("vna loss = %.4f.", vna_loss_meter.compute())
    logger.info("ft loss = %.4f.", ft_loss_meter.compute())
    logger.info("")


@hydra.main(config_path="configs/", version_base="1.3")
def main(cfg):
    """Main entrypoint for HiERO training on EgoClip."""

    logger.info("")
    logger.info("################################################################################################")
    logger.info("# HiERO: understanding the hierarchy of human behavior enhances reasoning on egocentric videos #")
    logger.info("################################################################################################")
    logger.info("")

    torch_rng = seed_everything(cfg.seed)

    # Step 1: Initialize the EgoClip/EgoMCQ datasets
    dset_train = hydra.utils.instantiate(cfg.train_dataset, features=cfg.features)
    dset_val = hydra.utils.instantiate(cfg.val_dataset, features=cfg.features)
    dl_train = build_dataloader(dset_train, cfg.batch_size, True, cfg.num_workers, drop_last=True, rng_generator=torch_rng)
    dl_val = build_dataloader(dset_val, 1, False, cfg.num_workers, drop_last=False, rng_generator=torch_rng)

    # Create infinite dataloader for training
    inf_dl_train: InfiniteLoader = iter(InfiniteLoader([dl_train]))

    # Step 2: Initialize model, task, optimizers and schedulers
    logger.info("Building the HiERO model...")
    model = hydra.utils.instantiate(cfg.model, input_size=dset_train.features.size, _recursive_=False).to(cfg.device)
    task = hydra.utils.instantiate(cfg.task, _recursive_=False).to(cfg.device)
    optimizer = build_optimizer([m for m in [model, task] if m is not None], cfg.optimizer)

    logger.info("Setting up the learning rate scheduler...")
    warmup_scheduler, scheduler = build_lr_scheduler(cfg.lr_warmup, cfg.num_epochs, cfg.lr_warmup_epochs, cfg.lr_min, dl_train, optimizer)

    # Step 3: Training on EgoClip
    max_steps = cfg.num_epochs * len(dl_train)

    for train_step in range(0, max_steps, cfg.steps_per_round):

        logger.info("")
        logger.info("Starting training step n. %3d/%3d...", train_step, cfg.num_epochs * len(dl_train))

        train(
            train_step,
            model,
            task,
            optimizer,
            warmup_scheduler,
            cfg.lr_warmup_epochs * len(dl_train) if cfg.lr_warmup else 0,
            scheduler,
            inf_dl_train,
            grad_clip=cfg.gradient_clip,
            num_steps=min(cfg.steps_per_round, max_steps - train_step),
            device=cfg.device,
        )

        if cfg.save_to is not None:
            path = f"{cfg.save_to}/model.pth"
            logger.info("Saving model to %s...", path)
            os.makedirs(cfg.save_to, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "task": task.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "warmup_scheduler": (warmup_scheduler.state_dict() if warmup_scheduler is not None else None),
                    "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                    "step": train_step + cfg.steps_per_round,
                    "config": cfg,
                },
                path,
            )
            logger.info("Hash: %s", compute_hash(path, "sha256"))

    # Step 4: Final validation on EgoMCQ
    validate(model, dl_val, task)


if __name__ == "__main__":
    main()  # pylint: disable=E1120
