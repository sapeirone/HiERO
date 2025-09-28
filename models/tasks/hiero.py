"""HiERO task."""

import itertools as it
import logging
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer

# CLIP tokenizer and text encoder from LaViLa
from models.ext.lavila.tokenizers import SimpleTokenizer
from models.ext.lavila.lavila import CLIP_TextEncoder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HiEROTask(nn.Module):
    """HiERO task."""

    def __init__(
        self,
        input_size: int,
        features_size: int,
        dropout: float = 0,
        loss_weight: float = 1.0,
        proc_loss_weight: float = 0.0,
        temperature: float = 0.05,
        # text encoder configuraiton
        text_encoder: Literal["distillbert", "clip"] = "distillbert",
        text_encoder_weights: Optional[str] = "",
        text_encoder_full_ft: Optional[bool] = False,
        # Parameters of the contrastive L_vna video-narrations alignment loss
        alpha: float = 1.0,
        beta: float = 1.0,
        contrastive_mode: Literal["v2t", "t2v", "sym"] = "sym",
        **kwargs,
    ):
        """HiERO task.

        Parameters
        ----------
        input_size : int
            Size of the input features.
        features_size : int
            Hidden size of the features.
        dropout : float, optional
            Dropout rate, by default 0.
        loss_weight : float, optional
            Loss weight, by default 1.0.
        proc_loss_weight : float, optional
            FT loss weight, by default 0.0.
        temperature : float, optional
            Contrastive loss temperature, by default 0.05.
        text_encoder : Literal["distillbert", "clip"], optional
            Text encoder modality, by default "distillbert".
        text_encoder_weights : Optional[str], optional
            Path to the text encoder weights, by default ""
        text_encoder_full_ft : Optional[bool], optional
            Whether to use full fine-tuning for the text encoder, by default False.
        alpha : float, optional
            Alpha parameter for the contrastive loss, by default 1.0.
        beta : float, optional
            Beta parameter for the contrastive loss, by default 1.0.
        contrastive_mode : Literal["v2t", "t2v", "sym"], optional
            Contrastive mode for the loss, by default "sym".
        """
        super().__init__()

        # Visual projection for the input features
        self.projector = Projection(input_size, features_size, dropout)

        # Alpha and beta parameters for L_vna
        self.alpha = alpha
        self.beta = beta
        # Contrastive alignment temperature
        self.temperature = temperature
        # L_vna contrastive alignment mode
        self.contrastive_mode = contrastive_mode

        # Weights of the contrastive alignment losses
        self.loss_weight = loss_weight
        self.proc_loss_weight = proc_loss_weight

        logger.info("")
        logger.info("Using HiERO contrastive loss (alpha=%.2f, beta=%.2f, temp=%.2f, mode=%s).", self.alpha, self.beta, self.temperature, self.contrastive_mode)
        logger.info("Loss weights: alignment=%.2f, procedural=%.2f.", self.loss_weight, self.proc_loss_weight)

        self.text_encoder = text_encoder
        if self.text_encoder == "distillbert":  # Used by Omnivore and EgoVLP variants
            self._build_distillbert_text_encoder(features_size)
        elif self.text_encoder == "clip":  # Used by LaViLa variant
            self._build_clip_text_encoder(features_size)

        # Load pretrained weights for the text model from the best EgoVLP checkpoint
        if text_encoder_weights:
            logger.info("Loading text encoder weights from %s...", text_encoder_weights)
            self.text_model.load_state_dict(torch.load(text_encoder_weights))

        if text_encoder_full_ft:
            logger.info("Using full fine-tuning for the text encoder.")
        self.text_model.requires_grad_(text_encoder_full_ft)

    def _build_distillbert_text_encoder(self, features_size: int):
        """Build the DistilBERT text encoder.

        Parameters
        ----------
        features_size : int
            Hidden features size.
        """
        # Step 1: build tokenizer for distilbert-base-uncased
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", TOKENIZERS_PARALLELISM=False)

        # Step 2: build model for distilbert-base-uncased
        self.text_model = AutoModel.from_pretrained("distilbert-base-uncased", cache_dir="pretrained/distilbert-base-uncased")

        # Step 3: build projection
        self.text_proj = nn.Sequential(nn.Linear(self.text_model.config.hidden_size, features_size), nn.ReLU(), nn.Linear(features_size, features_size))

    def _build_clip_text_encoder(self, features_size: int):
        """Build the CLIP text encoder.

        Parameters
        ----------
        features_size : int
            Hidden features size.
        """
        # Step 1: build tokenizer for the CLIP text encoder of LaViLa
        self.tokenizer = SimpleTokenizer()

        # Step 2: build model for LaViLa text encoder
        self.text_model = CLIP_TextEncoder(embed_dim=256, context_length=77, vocab_size=49408, transformer_heads=12, transformer_layers=12, transformer_width=768)

        # Step 3: build projection
        self.text_proj = nn.Sequential(nn.Linear(768, features_size), nn.ReLU(), nn.Linear(features_size, features_size))

    def forward(self, graphs: Data, data: Data, *args, **kwargs) -> torch.Tensor:
        """Forward features through the projection module.

        Parameters
        ----------
        graphs : torch.Tensor
            output of the temporal GNN
        data: Data
            input data

        Returns
        -------
        torch.Tensor
            output logits (tensor shape: [batch_size, n_classes])
        """

        return self.projector(graphs.x)

    def encode_text(self, narrations: list[str], device: torch.device = torch.device("cuda")) -> torch.Tensor:
        """Extract features for the textual narrations.

        Parameters
        ----------
        narrations : list[str]
            list of narrations
        device : torch.device, optional
            device, by default torch.device('cuda')

        Returns
        -------
        torch.Tensor
            text embeddings of the narrations
        """
        if self.text_encoder == "distillbert":
            tokens = self.tokenizer(narrations, return_tensors="pt", padding=True, truncation=True).to(device)
            text_embeddings = self.text_model(**tokens).last_hidden_state[:, 0, :]
        else:
            tokens = self.tokenizer(narrations).to(device)
            if len(narrations) == 1:
                tokens = tokens.unsqueeze(0)
            text_embeddings = self.text_model(tokens)

        return self.text_proj(text_embeddings)

    def compute_loss(self, vis_features: torch.Tensor, graphs: Data, data: Data) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the HiERO losses (L_vna and L_ft).

        Parameters
        ----------
        vis_features : torch.Tensor
            Output features of the nodes, projected using the task.
        graphs : Data
            Output graphs from the temporal backbone.
        data : Data
            Input data from the dataset.

        Returns
        -------
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The total loss and the individual losses.
        """

        # Extract features from the narrations
        narrations = list(it.chain.from_iterable(data.narrations))
        text_features = self.encode_text(narrations, vis_features.device)

        video_narrations_mask, pos, depth = graphs.video, graphs.pos, graphs.depth
        vis_features = vis_features[graphs.depth == 0]
        video_narrations_mask = graphs.video[graphs.depth == 0]
        pos = graphs.pos[graphs.depth == 0]
        depth = graphs.depth[graphs.depth == 0]

        # Create a mask to match videos with their narrations (should have shape num_video_samples x num_narrations)
        video_narrations_mask = video_narrations_mask[:, None] == data.narration_timestamps_batch[None, :]  # type: ignore

        # Video-Narrations Alignment loss (L_vna)
        vna_loss = compute_vna_loss(
            pos, depth, vis_features, text_features, data.narration_timestamps, video_narrations_mask, alpha=self.alpha, beta=self.beta, mode=self.contrastive_mode, temperature=self.temperature
        )

        # Functional Threads loss (L_ft)
        ft_loss = compute_ft_loss(graphs.x, graphs.video, graphs.depth, graphs.assignments, self.temperature)

        return self.loss_weight * vna_loss + self.proc_loss_weight * ft_loss, (vna_loss, ft_loss)


def compute_ft_loss(features: torch.Tensor, video: torch.Tensor, depth: torch.Tensor, assignments: torch.Tensor, temperature: float) -> torch.Tensor:
    """Compute the functional threads loss.

    Parameters
    ----------
    features : torch.Tensor
        Input features.
    video : torch.Tensor
        Video tensor mask.
    depth : torch.Tensor
        Depth mask.
    assignments : torch.Tensor
        Assignments mask.
    temperature : float
        Temperature parameter for the contrastive loss.

    Returns
    -------
    torch.Tensor
        Functional threads loss.
    """
    valid_samples = (depth > 0) & (assignments >= 0)
    features = features[valid_samples]
    features = nn.functional.normalize(features, p=2, dim=-1)
    sims = features @ features.T

    same_depth = depth[valid_samples, None] == depth[None, valid_samples]
    same_video = video[valid_samples, None] == video[None, valid_samples]
    same_assignment = assignments[valid_samples, None] == assignments[None, valid_samples]

    positives = same_depth & same_video & same_assignment
    negatives = (same_depth & same_video) & ~same_assignment

    sims = torch.where(positives | negatives, sims, -torch.inf)
    return -((sims / temperature).softmax(1) * positives).sum(1).log().mean()


class Projection(torch.nn.Module):
    """MLP-based projection."""

    def __init__(self, input_size: int, features_size: int = 1024, dropout: float = 0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, features_size),
            nn.LayerNorm(features_size),
            nn.ReLU(),
            nn.Linear(features_size, features_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_vna_loss(
    vis_features: torch.Tensor,
    text_features: torch.Tensor,
    pos: torch.Tensor,
    depth: torch.Tensor,
    narration_timestamps: torch.Tensor,
    video_narrations_mask: torch.Tensor,
    alpha: float = 1.0,
    beta: float = -1.0,
    mode: Literal["v2t", "t2v", "sym"] = "sym",
    temperature: float = 0.05,
) -> torch.Tensor:
    """Compute the Video-Narrations Alignment (VNA) loss.

    Parameters
    ----------
    vis_features : torch.Tensor
        Visual features of the nodes.
    text_features : torch.Tensor
        Text features of the nodes.
    pos : torch.Tensor
        Temporal timestamps of the nodes.
    depth : torch.Tensor
        Depth of the nodes.
    narration_timestamps : torch.Tensor
        Temporal timestamps of the narrations.
    video_narrations_mask : torch.Tensor
        Mask to identify valid video-narration pairs.
    alpha : float, optional
        Alpha parameter of the VNA loss, by default 1.0.
    beta : float, optional
        Beta parameter of the VNA loss, by default -1.0.
    mode : Literal["v2t", "t2v", "sym"], optional
        Mode of the VNA loss, by default "sym"
    temperature : float, optional
        Temperature parameter of the VNA loss, by default 0.05

    Returns
    -------
    torch.Tensor
        Video-Narrations Alignment loss.
    """

    # Normalize the input features
    vis_features = vis_features / vis_features.norm(dim=-1, p=2, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, p=2, keepdim=True)

    # Step 1: Intra-video positives
    extent_lower, extent_upper = pos - (2**depth) / alpha, pos + (2**depth) / alpha
    positives = (extent_lower[:, None] <= narration_timestamps[None, :]) & (narration_timestamps[None, :] <= extent_upper[:, None])
    positives = positives & video_narrations_mask

    # Step 2: Intra-video negatives
    if beta > 0:
        # Intra-video negatives (same video and within the temporal extent)
        extent_lower, extent_upper = pos - (2**depth) * (beta / alpha), pos + (2**depth) * (beta / alpha)
        negatives = (extent_lower[:, None] <= narration_timestamps[None, :]) & (narration_timestamps[None, :] <= extent_upper[:, None])
        negatives = (negatives & video_narrations_mask) | (~video_narrations_mask)
    else:
        negatives = torch.ones_like(positives).bool()

    # Step 3: Filter only valid samples (video segments with at least one positive narration and vice-versa)
    valid_narrations = positives.sum(0) > 0
    valid_segments = positives.sum(1) > 0

    vis_features, positives, negatives = vis_features[valid_segments], positives[valid_segments], negatives[valid_segments]
    text_features, positives, negatives = text_features[valid_narrations], positives[:, valid_narrations], negatives[:, valid_narrations]

    similarities = vis_features @ text_features.T

    # Step 4: Mask values that are neither positives nor negatives
    similarities = torch.where(positives | negatives, similarities, -torch.inf)

    loss_i = ((similarities / temperature).softmax(1) * positives).sum(1).log().mean()
    loss_j = ((similarities / temperature).T.softmax(1) * positives.T).sum(1).log().mean()

    if mode == "v2t":
        return -loss_i

    if mode == "t2v":
        return -loss_j

    return -(loss_i + loss_j)
