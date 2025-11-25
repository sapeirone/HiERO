"""EgoClip dataset."""

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
class EgoClipAction:
    """EgoClip narration with timestamp and textual description"""
    time: float
    narration: str


@dataclass
class EgoClipSegment:
    """EgoClip video segment with narrations"""
    video_uid: str
    part: int
    time_start: float
    actions: list[EgoClipAction]


class EgoClipDataset(Dataset):  # pylint: disable=W0223
    """EgoClip dataset."""

    def __init__(self, root: str, features: Features, skip_videos: Optional[str] = None):
        """Initialize the EgoClip dataset.

        This dataset supports a skip_videos list to exclude certain videos from the dataset.
        This can be used, for example, to avoid data leakage for downstream evaluation benchmarks.

        Parameters
        ----------
        root : str
            Dataset root.
        features : Features
            Features used in the dataset.
        skip_videos : Optional[str], optional
            Path to a file containing a list of videos to be excluded, by default None
        """
        super().__init__(root)

        logger.info("Loading EgoClip dataset...")

        self.features = features

        # List the existing features files
        self.existing_videos = [f.split(".")[0] for f in os.listdir(self._features_path) if f.endswith(".pt")]

        # Load the list of videos to skips, if any
        skip_videos = [f.strip() for f in open(skip_videos).readlines()] if skip_videos is not None else []
        self.load(skip_videos)

    @property
    def _features_path(self) -> str:
        """Get the path to the features directory.

        Returns
        -------
        str
            Path to the features directory.
        """
        return osp.join(self.raw_dir, "features", self.features.name)

    def load(self, skip_videos: Optional[List[str]]):
        """Load the EgoClip dataset, optionally skipping some of the videos.

        Parameters
        ----------
        skip_videos : Optional[List[str]]
            List of video uids to skip.
        """
        annotations = pd.read_csv(f"{self.root}/raw/annotations/egoclip.csv", sep="\t", on_bad_lines="skip")
        annotations = annotations.sort_values(by=["video_uid", "narration_time"])

        logger.info("Loaded %d annotations with %d unique videos.", len(annotations), len(annotations.video_uid.unique()))

        # Group annotations by video
        self.samples = []
        missing_videos = []
        for video_uid, video_annotations in tqdm(annotations.groupby("video_uid"), desc="Loading train data...", leave=False):
            if video_uid not in self.existing_videos:
                missing_videos.append(video_uid)
                continue

            if video_uid in skip_videos:
                continue

            feats = torch.load(osp.join(self._features_path, f"{video_uid}.pt"))
            video_duration = len(feats) * self.features.stride / self.features.fps

            # Split the video into 300s chunks (like EgoVLP)
            for part in range(int(video_duration // 300) + 1):

                time_start, time_end = part * 300, min((part + 1) * 300, video_duration)

                # Filter annotations for this part of the video
                part_ann = video_annotations[(time_start <= video_annotations.narration_time) & (video_annotations.narration_time < time_end)]

                actions = [EgoClipAction(float(narration_time), clip_text) for narration_time, clip_text in zip(part_ann.narration_time.tolist(), part_ann.clip_text.tolist())]

                if len(actions) == 0:
                    continue

                self.samples.append(EgoClipSegment(video_uid, part, time_start, actions))  # type: ignore

        # missing_annotations = annotations.video_uid.isin(missing_videos).sum()
        # logger.warning(f"{len(missing_videos)} out of {len(annotations.video_uid.unique())} videos were not found while initializing the EgoClip dataset!")
        # logger.warning(f"This corresponds to {missing_annotations} out of {len(annotations)} video-text pairs.")

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx):
        video: EgoClipSegment = self.samples[idx]
        # load features
        feats = torch.load(osp.join(self._features_path, f"{video.video_uid}.pt"), weights_only=True).float()

        idx_start = floor(video.time_start * self.features.fps // self.features.stride)
        idx_end = min(len(feats), ceil((video.time_start + 300) * self.features.fps // self.features.stride))

        feats = feats[idx_start:idx_end]

        return Data(
            video_uid=video.video_uid,
            x=feats.unsqueeze(1),
            mask=torch.ones(len(feats), dtype=torch.bool),
            indices=torch.arange(len(feats)),
            pos=(0.5 + torch.arange(len(feats)) + idx_start) * self.features.stride / self.features.fps,
            # Video total duration (in seconds)
            duration=(len(feats) * self.features.stride) / self.features.fps,
            # List of narrations and their respective timestamps
            narrations=[action.narration for action in video.actions],
            narration_timestamps=torch.Tensor([action.time for action in video.actions]),
        )
