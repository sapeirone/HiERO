"""A simple pipeline for video features extraction using Nvidia DALI"""

import os
from typing import Any, List

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.plugin import pytorch


@pipeline_def
def video_pipe(
    filenames: List[str],
    sequence_length: int,
    step: int,
    mean: tuple[float] = (0.485 * 255, 0.456 * 255, 0.406 * 255),
    std: tuple[float] = (0.229 * 255, 0.224 * 255, 0.225 * 255),
) -> Any:
    """The Nvidia DALI pipeline to load dense sequences of video frames.

    This pipeline loads videos from a list of file paths, with no shuffling.
    Videos are resized and cropped to 224x224 and normalized using
    the provided mean and std values.

    Parameters
    ----------
    filenames : List[str]
        List of video files to load.
    sequence_length : int
        Number of frames to load in each sequence.
    step : int
        Stride for loading frames.
    mean : List[float], optional
        Mean values for normalization, by default [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std : List[float], optional
        Standard deviation values for normalization, by default [0.229 * 255, 0.224 * 255, 0.225 * 255]

    Returns
    -------
    Any
        The Nvidia DALI pipeline.
    """
    videos = fn.readers.video_resize(
        device="gpu",
        filenames=filenames,
        sequence_length=sequence_length,
        shard_id=0,
        num_shards=1,
        random_shuffle=False,
        step=step,
        normalized=False,
        image_type=types.RGB,
        name="Reader",
        resize_shorter=224,
        skip_vfr_check=True,
        file_list_include_preceding_frame=False,
    )

    videos = fn.crop_mirror_normalize(
        videos,
        output_layout="CFHW",
        crop=(224, 224),
        mean=mean,
        std=std,
        mirror=0,
    )
    return videos


class DALILoader:
    """DALI Dataloader for the video loading pipeline"""

    def __init__(
        self,
        filenames: List[str],
        batch_size: int,
        sequence_length: int,
        step: int,
        mean: List[float] = [0.485 * 255, 0.456 * 255, 0.406 * 255],
        std: List[float] = [0.229 * 255, 0.224 * 255, 0.225 * 255],
    ):
        self.pipeline = video_pipe(
            batch_size=batch_size,
            sequence_length=sequence_length,
            step=step,
            num_threads=4,
            device_id=0,
            filenames=filenames,
            seed=123456,
            mean=mean,
            std=std,
        )
        self.pipeline.build()

        self.epoch_size = self.pipeline.epoch_size("Reader")
        self.dali_iterator = pytorch.DALIGenericIterator(
            self.pipeline,
            ["data"],
            reader_name="Reader",
            last_batch_policy=pytorch.LastBatchPolicy.PARTIAL,
            auto_reset=True,
            prepare_first_batch=True,
        )

    def __len__(self):
        return int(self.epoch_size)

    def __iter__(self):
        return self.dali_iterator.__iter__()
