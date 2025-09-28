"""
Step Grounding evaluation on Ego4D Goal-Step.
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
from typing import Callable, List, Literal, Optional

import hydra
import torch
from torch import nn
from torch_geometric.data import Data
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from ego4d_goalstep.utils.clusters import clusterize, compress
from ego4d_goalstep.utils.evaluate import display_results, evaluate_nlq_performance
from ego4d_goalstep.utils.utils import load_features
from utils.random import seed_everything


torch.set_grad_enabled(False)


def build_hiero_fe(model: nn.Module, task: nn.Module, stride: int = 16, fps: int = 30, device: torch.device = "cuda") -> Callable[[torch.Tensor], torch.Tensor]:
    """Build HiERO features extractor.

    Parameters
    ----------
    model : nn.Module
        HiERO temporal backbone.
    task : nn.Module
        HiERO task module.
    stride : int, optional
        Stride of the input features, by default 16.
    fps : int, optional
        Frames per second of the input, by default 30.
    device : torch.device, optional
        Device to run the model on, by default 'cuda'.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        Features extractor.
    """
    node_length = stride / fps

    def visual_fe(features: torch.Tensor):

        # Build the input structure
        pos = torch.arange(0, features.shape[0], device=device) * node_length
        indices = torch.arange(0, features.shape[0], device=device)
        batch = torch.zeros_like(pos, dtype=torch.long)
        mask = torch.ones_like(pos, dtype=torch.int).bool()
        data = Data(x=features.unsqueeze(1), pos=pos, indices=indices, batch=batch, mask=mask)

        # Forward through the hiero temporal backbone
        graphs = model(data.to(device=device))
        features = task(graphs, data)

        # Take features at the output of the decoder
        return features[graphs.depth == 0]

    def text_fe(text: List[str]):
        queries_features = task.encode_text(text)
        return nn.functional.normalize(queries_features, p=2, dim=-1)

    return visual_fe, text_fe


def build_encoders(
    features: Literal["omnivore_video_swinl", "egovlp"], ckpt: Optional[str], device: torch.device = "cuda"
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[List[str]], torch.Tensor]]:
    """Build visual and text encoders, possibily using HiERO.

    Three configurations are currently supported: egovlp, HiERO (omnivore) and HiERO (egovlp).

    Parameters
    ----------
    features : Literal['omnivore_video_swinl', 'egovlp']
        Input features.
    ckpt : Optional[str]
        Model checkpoint (for HiERO models).
    device : torch.device, optional
        Device to run the model on, by default 'cuda'.

    Returns
    -------
    tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[List[str]], torch.Tensor]]
        Visual and text encoders.
    """

    # Configuration 1: vanilla EgoVLP
    if features == "egovlp" and ckpt is None:
        print("Building vanilla EgoVLP video and text encoders...")

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", TOKENIZERS_PARALLELISM=False)
        text_model = AutoModel.from_pretrained("distilbert-base-uncased", cache_dir="pretrained/distilbert-base-uncased")
        text_model = text_model.to(device)
        text_proj = nn.Sequential(nn.ReLU(), nn.Linear(768, 256)).to(device)

        text_model.load_state_dict(torch.load("pretrained/egovlp_text.pth", weights_only=True), strict=True)
        text_proj.load_state_dict(torch.load("pretrained/egovlp_txt_proj.pth", weights_only=True), strict=True)

        text_model = text_model.eval()
        text_proj = text_proj.eval()

        def text_encoder_fe(text: List[str]):
            tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")
            queries = text_proj(text_model(**tokens).last_hidden_state[:, 0, :])
            return nn.functional.normalize(queries, p=2, dim=-1)

        return None, text_encoder_fe

    if features == "omnivore_video_swinl" and ckpt is None:
        print("Unsupported!!")

    # Configuration 2/3: HiERO with omnivore or egovlp features.
    print(f"Building HiERO with {features} features using checkpoint {ckpt}...")

    # Load configuration and weights from checkpoint
    state = torch.load(ckpt, weights_only=False)
    input_size = 1536 if "omnivore" in features else 256
    model = hydra.utils.instantiate(state["config"]["model"], clustering_at_inference="active", input_size=input_size, _recursive_=False).to(device)
    task = hydra.utils.instantiate(state["config"]["task"], _recursive_=False).to(device).eval()

    model.load_state_dict(state["model"], strict=True)
    task.load_state_dict(state["task"], strict=True)

    return build_hiero_fe(model, task)


def evaluate_grounding_step(
    annotations: dict,
    n_clusters: int,
    compression: int,
    text_encoder: Callable[[List[str]], torch.Tensor],
    visual_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    node_length: float = 16 / 30,
    features: Literal["omnivore-video-swinl", "egovlp"] = "egovlp",
):
    """Evaluate the Step Grounding task.

    Parameters
    ----------
    annotations : dict
        Ego4d Goal-Step annotations.
    n_clusters : int
        Number of clusters to find.
    compression : int
        Compression factor for cluster assignments aggregation.
    text_encoder : Callable[[List[str]], torch.Tensor]
        Text encoder.
    visual_encoder : Optional[Callable[[torch.Tensor], torch.Tensor]], optional
        Visual encoder, by default None.
    node_length : float, optional
        Node length in seconds, by default 16/30.
    features : Literal["omnivore-video-swinl", "egovlp"], optional
        Features to use, by default "egovlp".
    """
    predictions = []

    for video in tqdm(annotations, leave=False):
        # For each video in the annotations
        video_uid = video["video_uid"].replace("grp-", "")

        video_features = load_features(video_uid, root=f"data/ego4d/raw/features/{features}")
        if visual_encoder is not None:
            # Using HiERO
            video_features = visual_encoder(video_features)

        # Clusterize features
        clusters = compress(clusterize(video_features, n_clusters))
        clusters = [(start, length) for (start, length) in clusters if length > compression]

        # Aggregated features from the same cluster
        feature_clusters = [video_features[start : start + length].mean(0) for (start, length) in clusters]
        feature_clusters = torch.stack(feature_clusters)
        feature_clusters = nn.functional.normalize(feature_clusters, p=2, dim=-1)

        for clip in video["clips"]:
            # For each video in the annotations
            clip_start = clip["video_start_sec"]

            for ann_datum in clip["annotations"]:
                annotation_uid = ann_datum["annotation_uid"]

                # Extract the NLQ-like queries corresponding to the steps
                queries = [f"#C C {q['query']}" for q in ann_datum["language_queries"]]
                queries_features: torch.Tensor = text_encoder(queries)

                similarities = queries_features @ feature_clusters.T

                # For each query, find the top 5 most confident clusters
                for idx, _ in enumerate(queries):
                    top5_similarities_per_query = similarities[idx].topk(min(5, len(similarities[idx])), largest=True).indices

                    predictions.append(
                        {
                            "clip_uid": clip["clip_uid"],
                            "annotation_uid": annotation_uid,
                            "query_idx": idx,
                            "predicted_times": [
                                [
                                    clusters[i][0] * node_length - clip_start,
                                    (clusters[i][0] + clusters[i][1]) * node_length - clip_start,
                                ]
                                for i in top5_similarities_per_query
                            ],
                        }
                    )

    thresholds = [0.3, 0.5, 0.01]
    topk = [1, 3, 5]

    results, mIoU = evaluate_nlq_performance(predictions, {"videos": annotations}, thresholds, topk, per_instance=False)
    print(display_results(results, mIoU, thresholds, topk))


def main(arg):
    """Run validation on Ego4d Goal-Step Step Grounding."""

    seed_everything(arg.seed)

    # Build the visual and text encoders
    visual_encoder, text_encoder = build_encoders(arg.features, arg.ckpt)

    annotations = json.load(open("ego4d_goalstep/annotations/val.json", "r", encoding="utf-8"))

    print("Starting evaluation on Ego4d Goal-Step Step Grounding...")
    evaluate_grounding_step(
        annotations["videos"],
        n_clusters=arg.n_clusters,
        compression=arg.threshold,
        text_encoder=text_encoder,
        visual_encoder=visual_encoder,
        node_length=16 / 30,
        features=arg.features,
    )

    print("")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Ego4d Goal-Step Step Grounding evaluation.")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--features", type=str, choices=["omnivore_video_swinl", "egovlp"], required=True)
    parser.add_argument("--ckpt", type=str)

    # Parameters for spectral clustering based procedure learning
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--threshold", type=int, default=4)

    main(parser.parse_args())
