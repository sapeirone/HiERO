"""Main entrypoint for HiERO validation on EgoMCQ."""

import logging
import os
import warnings

import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torchmetrics.aggregation import MeanMetric
from tqdm.auto import tqdm

from models.hiero import HiERO
from models.tasks.hiero import HiEROTask

from utils import build_dataloader, compute_hash, seed_everything

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


warnings.filterwarnings("ignore")


def resume_from(path: str, device: torch.device = torch.device("cuda")) -> tuple[OmegaConf, dict, dict]:
    """Resume configuration and model weights from checkpoint.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.
    device : torch.device, optional
        Device to load the model on, by default torch.device("cuda")

    Returns
    -------
    tuple(OmegaConf, dict, dict)
        Model configuration and weights.
    """

    if path.endswith(".latest"):
        basename = os.path.basename(path)
        path = os.path.join(
            os.path.dirname(path),
            sorted([f for f in os.listdir(os.path.dirname(path)) if f.startswith(basename) and f.endswith(".pth")])[-1],
        )
    logger.info("Resuming from %s...", path)
    logger.debug("Hash: %s", compute_hash(path, "sha256"))
    logger.info("")

    state = torch.load(path, map_location=device, weights_only=False)

    return OmegaConf.create(state["config"]), state["model"], state["task"]


@torch.no_grad()
def validate(model: HiERO, dataloader: DataLoader, task: HiEROTask, device: torch.device = torch.device("cuda")):
    """Run a validation step on EgoMCQ.

    Parameters
    ----------
    model : HiERO
        temporal model (e.g. TRN or GraphUNet)
    dataloader : DataLoader
        dataloader over validation data
    task : HiEROTask
        main task
    device : torch.device
        Torch device to run on
    """

    # Put models in eval mode
    model.eval()
    task.eval()

    inter_meter, intra_meter = MeanMetric().to(device=device), MeanMetric().to(device=device)

    pbar = tqdm(dataloader, desc="Validating HiERO on EgoMCQ...", leave=False)
    for data in pbar:

        choices = Batch.from_data_list(data["choices"]).to(device=device)  # type: ignore

        # Compute the output graphs from HiERO
        graphs = model(choices)
        outputs = task(graphs, data)

        # For EgoMCQ, we take the output graphs of the model (end of the decoder)
        # and slice them to take only the central node for each sample
        outputs = outputs[graphs.depth == 0]
        visual_features = torch.stack([outputs[choices.batch == idx][(choices.batch == idx).sum() // 2] for idx in torch.unique(choices.batch)])
        text_options = task.encode_text(data["query"], device=device)  # type: ignore

        pred = nn.functional.cosine_similarity(visual_features, text_options).squeeze().argmax(dim=-1).item()  # pylint: disable=E1102

        if int(data["type"]) == 1:  # 1 for inter; 2 for intra
            inter_meter.update(pred == data["answer"].item())  # type: ignore
        else:
            intra_meter.update(pred == data["answer"].item())  # type: ignore

        pbar.set_description(f"EgoMCQ Accuracy: intra = {100 * intra_meter.compute():.3f}, inter = {100 * inter_meter.compute():.3f}.")

    logger.info("EgoMCQ Accuracy: intra = %.3f, inter = %.2f.", 100 * intra_meter.compute(), 100 * inter_meter.compute())


@hydra.main(config_path="configs/", version_base="1.3")
def main(cfg):
    """Main entrypoint for HiERO validation on EgoMCQ."""

    logger.info("")
    logger.info("################################################################################################")
    logger.info("# HiERO: understanding the hierarchy of human behavior enhances reasoning on egocentric videos #")
    logger.info("################################################################################################")
    logger.info("")

    # Step 1: Resume training from a previous checkpoint
    ckpt_cfg, model_state, task_state = resume_from(cfg.resume_from, device=cfg.device)

    torch_rng = seed_everything(cfg.seed)

    # Step 2: Initialize the EgoClip/EgoMCQ datasets
    dset_val = hydra.utils.instantiate(cfg.val_dataset, features=cfg.features)
    dl_val = build_dataloader(dset_val, 1, False, cfg.num_workers, drop_last=False, rng_generator=torch_rng)

    # Step 3: Initialize model and task
    model = hydra.utils.instantiate(ckpt_cfg.model, input_size=dset_val.features.size, _recursive_=False).to(cfg.device)
    task = hydra.utils.instantiate(ckpt_cfg.task, _recursive_=False).to(cfg.device)

    model.load_state_dict(model_state)
    task.load_state_dict(task_state)

    logger.info("")
    logger.info("Starting validation on EgoMCQ...")

    # Step 4: Validation on EgoMCQ
    validate(model, dl_val, task)


if __name__ == "__main__":
    main()  # pylint: disable=E1120
