#######################
# Features extraction #
#######################

import logging
import math
import os.path as osp
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import hydra
import torch
from pipe import DALILoader
from tqdm.auto import tqdm

torch.set_grad_enabled(False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_video(path: str, model: torch.nn.Module, output_path: str, batch_size: int, num_frames: int, stride: int, device: torch.device):
    """Extract features from a video file.

    Parameters
    ----------
    path : str
        Path to the video file.
    model : torch.nn.Module
        The feature extraction model.
    output_path : str
        Path to save the extracted features.
    batch_size : int
        Number of samples per batch.
    num_frames : int
        Number of frames to process.
    stride : int
        Stride for frame extraction.
    device : torch.device
        Device to run the model on.
    """
    dataloader = DALILoader([path], batch_size=batch_size, sequence_length=num_frames, step=stride)

    features = []
    for data in tqdm(dataloader, desc=f"Extracting features from {osp.basename(path)}...", total=math.ceil(len(dataloader) / batch_size)):
        features.append(model(data[0]["data"].to(device)).cpu())

    features = torch.cat(features)
    torch.save(features, output_path)


@hydra.main(config_path="configs/", config_name="defaults", version_base="1.3")
def main(cfg):
    logger.info("Starting feature extraction...")

    # Build the features extraction model
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(cfg.device)
    model = model.eval()

    process_video(
        path=cfg.path,
        model=model,
        output_path=osp.join(cfg.out_dir, osp.basename(cfg.path).replace(".mp4", ".pt")),
        batch_size=cfg.batch_size,
        num_frames=cfg.num_frames,
        stride=cfg.stride,
        device=cfg.device,
    )


if __name__ == "__main__":
    main()
