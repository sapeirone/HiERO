"""Utility functions for features loading"""

import pandas as pd
import torch

groups = pd.read_csv('ego4d_goalstep/annotations/goalstep_video_groups.tsv', sep='\t')
groups = [eval(group) for group in groups.video_group.tolist()]
groups = {group[0]: group for group in groups}


def load_features(video_uid: str, root: str, device: torch.device = "cuda") -> torch.Tensor:
    """Load video features for videos or video groups"""
    if video_uid in groups:
        return torch.cat([torch.load(f"{root}/{v}.pt", weights_only=True) for v in groups[video_uid]]).to(device)

    return torch.load(f"{root}/{video_uid}.pt", weights_only=True).to(device)
