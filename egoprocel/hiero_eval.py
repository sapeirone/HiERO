"""
Evaluate Omnivore and EgoVLP HiERO models on the EgoProceL task.
"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import os
import os.path as osp
from typing import Any, Dict, Callable

import hydra
import numpy as np
import torch
from torch_geometric.data import Data

from egoprocel.configs import DATASETS
from egoprocel.evaluate import eval_egoprocel, sample_labels
from egoprocel.utils import clusterize, get_fps

torch.set_grad_enabled(False)

FEAT_STRIDE = 16
SAMPLING_RATE = 2

DEVICE = "cuda"


def build_hiero_model(ckpt: str, fps: float, depth: int, use_proj_head: bool, input_size: int, stride: int = FEAT_STRIDE, device: str = DEVICE) -> Callable[[torch.Tensor], torch.Tensor]:
    """Load a HiERO model from a checkpoint.

    This function returns a callable that takes some input features (e.g., Omnivore or EgoVLP) and processes them using HiERO.

    Parameters
    ----------
    ckpt : str
        The path to the HiERO checkpoint.
    fps : float
        The frames per second of the input video.
    depth : int
        The depth of the HiERO model.
    use_proj_head : bool
        Whether to use the language-aligned projection head of the HiERO model.
    input_size : int
        The size of the input features.
    stride : int, optional
        The stride of the input features, by default FEAT_STRIDE
    device : str, optional
        The device to run the model on, by default DEVICE

    Returns
    -------
    Callable[torch.Tensor, [torch.Tensor]]
        A wrapper for the HiERO model that takes the raw input features and processes them using the HiERO backbone.
    """
    weights = torch.load(ckpt, weights_only=False)

    # Instantiate the model
    model = hydra.utils.instantiate(weights["config"]["model"], clustering_at_inference="active", input_size=input_size, _recursive_=False).to(device)
    task = hydra.utils.instantiate(weights["config"]["task"], _recursive_=False).to(device)

    model.load_state_dict(weights["model"], strict=True)
    task.load_state_dict(weights["task"], strict=True)

    model = model.eval()
    task = task.eval()

    node_length = (stride / fps) * (2**depth)

    def features_extractor(features: torch.Tensor):
        # Build the input structure
        pos = torch.arange(0, features.shape[0], device=device) * node_length
        indices = torch.arange(0, features.shape[0], device=device)
        batch = torch.zeros_like(pos, dtype=torch.long)
        mask = torch.ones_like(pos, dtype=torch.int).bool()
        data = Data(x=features.unsqueeze(1), pos=pos, indices=indices, batch=batch, mask=mask)

        # Forward through the temporal model
        graphs = model(data.to(device=device))

        # Optionally use the projection head
        if use_proj_head:
            features = task(graphs, data)
        else:
            features = graphs.x

        # Take only the nodes at a given depth in the decoder
        return features[graphs.depth == depth]

    return features_extractor


def eval_task(dset: str, ckpt: str, depth: int, use_proj_head: bool, ann_root: str, feat_root: str, videos_root: str, num_keysteps: int, temp: float, n: Any):
    """Run Procedure Learning validation on one single task from EgoProceL.

    Parameters
    ----------
    dset : str
        The name of the dataset.
    ckpt : str
        The path to the HiERO checkpoint.
    depth : int
        The layer of the decoder from which the output features are taken.
    use_proj_head : bool
        Whether to use the language-aligned projection head.
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
            print("Missing features for video", f"{feat_root}/{video}.pt")
            continue

        fps = get_fps(f"{videos_root}/{video}.mp4")
        input_size = 256 if "egovlp" in feat_root else 1536
        model = build_hiero_model(ckpt, fps=fps, depth=depth, use_proj_head=use_proj_head, input_size=input_size)

        *_, labels = sample_labels(dset, annotation_path=f"{ann_root}/{video}.csv", video_path=f"{videos_root}/{video}.mp4")
        features = torch.load(f"{feat_root}/{video}.pt", weights_only=True)
        features = model(features)

        # Clusterize features using spectral clustering, padding if necessary
        clusters = clusterize(features, n, temp)
        clusters = torch.repeat_interleave(torch.from_numpy(clusters), (FEAT_STRIDE // SAMPLING_RATE) * (2**depth))
        clusters = torch.cat([clusters, torch.zeros(max(0, labels.shape[0] - clusters.shape[0]))])
        clusters = clusters[: labels.shape[0]].numpy()

        # disc_gt contains the discovered ground truth labels
        video_rec, video_iou, video_prec, *_ = eval_egoprocel(clusters, labels, n_keystep=num_keysteps + 1, M=n, per_keystep=True)

        avg_iou.append(video_iou)
        avg_rec.append(video_rec)
        avg_prec.append(video_prec)

    return np.mean(avg_rec), np.mean(avg_prec), np.mean(avg_iou)


def eval_multitask(
    dset_name: str,
    dset_config: Dict[str, Any],
    ckpt: str,
    depth: int,
    use_proj_head: bool,
    ann_root: str,
    feat_root: str,
    videos_root: str,
    temp: float,
    n: Any,
):
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

    for task in dset_config["tasks"]:

        rec, prec, iou = eval_task(
            dset_name,
            ckpt,
            depth,
            use_proj_head,
            ann_root=osp.join(ann_root, task["annotations"]),
            feat_root=os.path.join(feat_root, dset_config["features"]),
            videos_root=osp.join(videos_root, task["video"]),
            num_keysteps=task["num_keysteps"],
            temp=temp,
            n=n,
        )

        avg_rec.append(rec)
        avg_prec.append(prec)
        avg_iou.append(iou)

    return np.mean(avg_rec), np.mean(avg_prec), np.mean(avg_iou)


def main(args):
    # Log the strategy used to find the number of clusters
    dsets = list(DATASETS.keys()) if args.dset == "all" else args.dset

    print(f"Using fixed number of {args.n} clusters with temperature {args.temp}.")

    features_root = osp.join(args.features_root, args.features)
    print(f"Features root: {features_root}")
    print(f"Videos root: {args.videos_root}")
    print(f"Annotations root: {args.annotations_root}")

    print(f"Resuming checkpoint from {args.ckpt}...")

    avg_f1, avg_iou = [], []

    for dset in dsets:

        print(f"Starting evaluation on dataset {dset}...", end=" ")

        if "tasks" in DATASETS[dset]:
            rec, prec, iou = eval_multitask(
                dset,
                DATASETS[dset],
                ckpt=args.ckpt,
                depth=args.depth,
                use_proj_head=args.use_proj_head,
                feat_root=features_root,
                ann_root=args.annotations_root,
                videos_root=args.videos_root,
                temp=args.temp,
                n=args.n,
            )
        else:
            task = DATASETS[dset]
            rec, prec, iou = eval_task(
                dset,
                ckpt=args.ckpt,
                depth=args.depth,
                use_proj_head=args.use_proj_head,
                feat_root=osp.join(features_root, task["features"]),
                ann_root=osp.join(args.annotations_root, task["annotations"]),
                videos_root=osp.join(args.videos_root, task["video"]),
                num_keysteps=task["num_keysteps"],
                temp=args.temp,
                n=args.n,
            )

        f1 = (2 * rec * prec) / (prec + rec)
        print(f"F1 = {100 * f1:.2f}, IoU = {100 * iou:.2f}.")

        avg_f1.append(f1)
        avg_iou.append(iou)

    print("")
    print(f"Overall metrics: F1 = {100 * np.mean(avg_f1):.2f}, IoU = {100 * np.mean(avg_iou):.2f}.")
    print("")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dset", type=str, default="all", choices=[*list(DATASETS.keys()), "all"], nargs="+")

    arg_parser.add_argument("--ckpt", type=str, required=True)
    arg_parser.add_argument("--depth", type=int, default=2)
    arg_parser.add_argument("--use-proj-head", action="store_true", default=False)

    arg_parser.add_argument("--device", type=str, default="cuda")

    arg_parser.add_argument("--temp", type=float, default=0.5)
    arg_parser.add_argument("--n", default=7)  # integer, "oracle", "adaptive", "stsc"

    arg_parser.add_argument("--annotations-root", type=str, default="egoprocel/annotations")
    arg_parser.add_argument("--features-root", type=str, default="egoprocel/features")
    arg_parser.add_argument("--videos-root", type=str, default="egoprocel/videos")

    arg_parser.add_argument("--features", type=str, choices=["egovlp", "omnivore_video_swinl_window-16"], default="egovlp")

    main(arg_parser.parse_args())
