"""
Evaluate Omnivore and EgoVLP baselines on the EgoProceL task.
"""

import argparse
import os
import os.path as osp
from typing import Any, Dict

import numpy as np
import torch

from egoprocel.configs import DATASETS
from egoprocel.evaluate import eval_egoprocel, sample_labels
from egoprocel.utils import clusterize

torch.set_grad_enabled(False)


FEATURES_STRIDE = 16
SAMPLING_RATE = 2

DEVICE = "cuda"


def eval_task(dset: str, ann_root: str, feat_root: str, videos_root: str, num_keysteps: int, temp: float, n: Any) -> tuple[float, float, float]:
    """Run Procedure Learning validation on one single task from EgoProceL.

    Parameters
    ----------
    dset : str
        The name of the dataset.
    ann_root : str
        The root path to the annotations.
    feat_root : str
        The root path to the features.
    videos_root : str
        The root path to the videos.
    num_keysteps : int
        The number of keysteps.
    temp : float
        The temperature parameter in the spectral clustering algorithm.
    n : Any
        The number of clusters.

    Returns
    -------
    tuple[float, float, float]
        The average recall, precision, and IoU.
    """

    avg_iou, avg_rec, avg_prec = [], [], []

    for video in os.listdir(ann_root):
        video = video.replace(".csv", "")

        if not os.path.exists(f"{feat_root}/{video}.pt"):
            print("Missing features for video %s", f"{feat_root}/{video}.pt")
            continue

        *_, labels = sample_labels(dset, annotation_path=f"{ann_root}/{video}.csv", video_path=f"{videos_root}/{video}.mp4")
        features = torch.load(f"{feat_root}/{video}.pt", weights_only=True)

        # Clusterize features using spectral clustering, padding if necessary
        features = features[2::4]
        clusters = clusterize(features, n, temp)
        clusters = torch.repeat_interleave(torch.from_numpy(clusters), FEATURES_STRIDE // SAMPLING_RATE * 4)
        clusters = torch.cat([clusters, torch.zeros(max(0, labels.shape[0] - clusters.shape[0]))])
        clusters = clusters[: labels.shape[0]].numpy()

        # Compute the metrics for the current video
        # Here (num_keysteps + 1) is used to account for the background pseudo-class
        video_rec, video_iou, video_prec, *_ = eval_egoprocel(clusters, labels, n_keystep=num_keysteps + 1, M=n, per_keystep=True)

        avg_iou.append(video_iou)
        avg_rec.append(video_rec)
        avg_prec.append(video_prec)

    return np.mean(avg_rec), np.mean(avg_prec), np.mean(avg_iou)


def eval_multitask(dset: str, dset_config: Dict[str, Any], ann_root: str, feat_root: str, videos_root: str, temp: float, n: Any) -> tuple[float, float, float]:
    """Run Procedure Learning evaluation on dataset from EgoProceL containing multiple tasks (e.g., CMU).

    Parameters
    ----------
    dset : str
        The name of the dataset.
    dset_config : Dict[str, Any]
        The dataset config, including the list of tasks.
    ann_root : str
        The root path to the annotations.
    feat_root : str
        The root path to the features.
    videos_root : str
        The root path to the videos.
    num_keysteps : int
        The number of keysteps.
    temp : float
        The temperature parameter in the spectral clustering algorithm.
    n : Any
        The number of clusters.

    Returns
    -------
    tuple[float, float, float]
        The average recall, precision, and IoU.
    """

    avg_rec, avg_prec, avg_iou = [], [], []

    for task in dset_config['tasks']:

        rec, prec, iou = eval_task(dset, ann_root=osp.join(ann_root, task["annotations"]), 
                                   feat_root=os.path.join(feat_root, dset_config["features"]), 
                                   videos_root=osp.join(videos_root, task["video"]), 
                                   num_keysteps=task["num_keysteps"], temp=temp, n=n)

        avg_rec.append(rec)
        avg_prec.append(prec)
        avg_iou.append(iou)

    return np.mean(avg_rec), np.mean(avg_prec), np.mean(avg_iou)


def main(args):
    dsets = list(DATASETS.keys()) if args.dset == "all" else args.dset

    print(f"Using fixed number of {args.n} clusters with temperature {args.temp}.")

    print("")
    print(f"Features root: {osp.join(args.features_root, args.features)}")
    print(f"Videos root: {args.videos_root}")
    print(f"Annotations root: {args.annotations_root}")
    print("")

    # Compute the average F1 and IoU across all EgoProceL tasks
    avg_f1, avg_iou = [], []

    for dset in dsets:
        print(f"Starting evaluation on dataset {dset}... ", end=" ")

        if "tasks" in DATASETS[dset]:
            rec, prec, iou = eval_multitask(dset, DATASETS[dset],
                feat_root=osp.join(args.features_root, args.features),
                ann_root=args.annotations_root,
                videos_root=args.videos_root,
                temp=args.temp,
                n=args.n,
            )
        else:
            task = DATASETS[dset]
            rec, prec, iou = eval_task(
                dset,
                feat_root=osp.join(args.features_root, args.features, task["features"]),
                ann_root=osp.join(args.annotations_root, task["annotations"]),
                videos_root=osp.join(args.videos_root, task["video"]),
                num_keysteps=task["num_keysteps"],
                temp=args.temp,
                n=args.n,
            )

        task_f1 = (2 * rec * prec) / (prec + rec)

        print(f"F1 = {100 * task_f1:.2f}, IoU = {100 * iou:.2f}.")

        avg_f1.append(task_f1)
        avg_iou.append(iou)

    print("")
    print(f"Overall metrics: F1 = {100 * np.mean(avg_f1):.2f}, IoU = {100 * np.mean(avg_iou):.2f}.")
    print("")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dset", type=str, default="all", choices=[*list(DATASETS.keys()), "all"], nargs="+")

    arg_parser.add_argument("--device", type=str, default="cuda")

    arg_parser.add_argument("--temp", type=float, default=0.5)
    arg_parser.add_argument("--n", default=7, type=int)

    arg_parser.add_argument("--annotations-root", type=str, default="egoprocel/annotations")
    arg_parser.add_argument("--features-root", type=str, default="egoprocel/features")
    arg_parser.add_argument("--videos-root", type=str, default="egoprocel/videos")

    arg_parser.add_argument("--features", type=str, choices=["egovlp", "omnivore_video_swinl_window-16"], default="egovlp")

    main(arg_parser.parse_args())
