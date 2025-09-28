"""This file defines the base class for EgoVLP."""

import logging

import torch

from einops import rearrange

from models.egovlp.model import FrozenInTime as EgoVLP

logger = logging.getLogger(__name__)


class EgoVLPFeaturesExtractor(torch.nn.Module):
    """Wrapper for the EgoVLP backbone."""

    def __init__(
        self,
        num_frames: int,
        checkpoint_path: str = "../pretrained/egovlp.pth",
        proj: bool = False,
        vit_ckpt: str = "../pretrained/jx_vit_base_p16_224-80ecf9dd.pth",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        video_params = {"model": "SpaceTimeTransformer", "arch_config": "base_patch16_224", "num_frames": num_frames, "pretrained": True, "time_init": "zeros", "vit_ckpt": vit_ckpt}
        text_params = {"model": "distilbert-base-uncased", "pretrained": True, "input": "text"}

        self.net = EgoVLP(video_params, text_params, projection_dim=256, load_checkpoint=checkpoint_path).to("cuda")

        if not proj:
            self.net.txt_proj = torch.nn.Identity().to("cuda")
            self.net.vid_proj = torch.nn.Identity().to("cuda")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Omnivore features extractor.

        Parameters
        ----------
        x : torch.Tensor
            Input frames.

        Returns
        -------
        torch.Tensor
            Extracted features.
        """

        frames = rearrange(x, "b c t h w -> b t c h w")

        return self.net.compute_video(frames.contiguous())
