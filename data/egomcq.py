import json
import logging
import os
import os.path as osp
from dataclasses import dataclass
from math import ceil, floor
from typing import List, Optional

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from tqdm.auto import tqdm

from data.features import Features

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class EgoMCQChoice:
    video_uid: str
    start: float
    end: float


@dataclass
class EgoMCQSample:
    text: str
    choices: List[EgoMCQChoice]
    answer: int
    type: int


class EgoMCQDataset(Dataset):
    """EgoMCQ dataset."""

    def __init__(self, root: str, features: Features, context_window_size: int = 4):
        super().__init__(root)

        self.features = features
        self.context_window_size = context_window_size

        self.samples = self.load_data()

    @property
    def _features_path(self) -> str:
        """Get the path to the features directory.

        Returns
        -------
        str
            path to the features directory
        """
        return osp.join(self.raw_dir, "features", self.features.name)

    def load_data(self):
        annotations = json.load(open(osp.join(self.raw_dir, "annotations/egomcq.json"), 'r'))
        
        # When using features extracted by us, a very small number of videos is absent
        # since these features were extracted from Ego4Dv2 which dropped some very small videos.
        existing_videos = [f.split('.')[0] for f in os.listdir(self._features_path) if f.endswith('.pt')]

        samples = []
        missing = 0
        for sample in annotations.values():
            query = sample["query"]

            choices = []
            for choice in sample["choices"].values():
                choices.append(EgoMCQChoice(choice["video_uid"], float(choice["clip_start"]), float(choice["clip_end"])))
                
            if any([choice.video_uid not in existing_videos for choice in choices]):
                missing += 1
                continue

            samples.append(EgoMCQSample(query["clip_text"], choices, sample["answer"], sample["types"]))

        if missing > 0:
            logger.debug("%d out of %d were not found.", missing, len(annotations))
        return samples

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx):
        sample: EgoMCQSample = self.samples[idx]

        choices = []

        for choice in sample.choices:
            # load features
            feats = torch.load(osp.join(self._features_path, f"{choice.video_uid}.pt"))

            segment_idx_start = floor(choice.start * self.features.fps // self.features.stride)
            segment_idx_end = ceil(choice.end * self.features.fps // self.features.stride)

            # Make sure not to overshoot the video length
            segment_idx_end = max(segment_idx_end, segment_idx_start + 1)  # and that the clip has at least one segment
            segment_idx_end = min(segment_idx_end, len(feats))

            # Find the maximum size of a window around the clip
            available_window = min(segment_idx_start, len(feats) - segment_idx_end)
            available_window = min(available_window, self.context_window_size)

            # Compute the start and end indices of the segment
            window_idx_start = segment_idx_start - available_window
            window_idx_end = segment_idx_end + available_window
            
            if window_idx_start == window_idx_end:
                # degenerate case, just take one segment
                feats = torch.zeros((1, feats.shape[1]))
            else:
                feats = feats[window_idx_start:window_idx_end]

            choices.append(
                Data(
                    video_uid=choice.video_uid,
                    x=feats.unsqueeze(1),
                    mask=torch.ones(len(feats), dtype=torch.bool),
                    indices=torch.arange(len(feats)),
                    pos=(0.5 + torch.arange(len(feats))) * self.features.stride / self.features.fps,
                    fps=self.features.fps,
                    # Stride between feature vectors
                    feat_stride=self.features.stride,
                    # Number of frames / feature vector
                    feat_num_frames=self.features.window,
                    # Video total duration (in seconds)
                    duration=(len(feats) * self.features.stride) / self.features.fps,
                    # List of narrations and their respective timestamps
                )
            )

        return {"query": sample.text, "choices": choices, "answer": sample.answer, "type": sample.type}
