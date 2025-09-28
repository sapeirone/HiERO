# pylint: disable=C0103

"""HiERO temporal backbone."""

import logging

from typing import Tuple, Literal, Optional, Any, List

import hydra

import torch
from torch import nn, Tensor

import torch_geometric
from torch_geometric import nn as gnn
from torch_geometric.data import Data, Batch
from torch_geometric.nn.unpool import knn_interpolate

from einops.layers.torch import Rearrange

from einops import rearrange
from torch_kmeans import KMeans


logger = logging.getLogger(__name__)


class HiERO(torch.nn.Module):
    """HiERO temporal backbone."""

    def __init__(
        self,
        input_size: int,
        conv: dict,
        k: float = 2.0,
        n_layers: int = 2,
        hidden_size: int = 256,
        depth: int = 3,
        input_projection: Optional[bool] = False,
        dropout: float = 0,
        pool: Literal["batch_subsampling", "video_subsampling", "max", "mean"] = "batch_subsampling",
        # HiERO clustering parameters
        n_clusters: int = 8,
        clustering_sample_points: int = 32,
        clustering_at_inference: bool = False,
        **kwargs,
    ):
        """Build a Graph UNet for HiERO.

        HiERO is built as an encoder-decoder architecture inspired by Graph U-Net.
        The two branches share the same components but serve different roles.
        The Temporal Encoder implements local temporal reasoning, hierarchically aggregating
        information between temporally close segments, while the Function-Aware Decoder extends
        temporal reasoning to nodes that may be temporally distant but functionally similar,
        by connecting nodes belonging to the same thread.

        Parameters
        ----------
        input_size : int
            Number of input features.
        conv: dict
            Configuration for the GNN layers.
        k: float
            Radius for estimating the connectivity of the input graph.
        n_layers: int
            Number of graph conv modules at each encoder/decoder layer.
        hidden_size : int, optional
            Number of hidden features, by default 1024.
        depth : int, optional
            Depth of the unet architecture, by default 3.
        input_projection: Optional, bool
            Use an input projection.
        dropout : float, optional
            Dropout in the GNN layers, by default 0.1.
        pool: Literal['batch_subsampling', 'video_subsampling', 'max', 'mean']
            Pooling method for the temporal dimension.
        n_clusters : int
            Number of clusters for the HiERO clustering in the decoder.
        clustering_sample_points : int
            Number of sample points for the subsampling the graph during clustering.
        clustering_at_inference : bool
            Use HiERO clustering at inference time.
        """
        super().__init__(**kwargs)

        self.k = k
        self.hidden_size = hidden_size

        self.pool = pool

        # Parameters for HiERO clustering
        self.n_clusters = n_clusters
        self.clustering_sample_points = clustering_sample_points
        self.clustering_at_inference = clustering_at_inference

        logger.info("")
        logger.info("Initializing HiERO model with config: input_size=%d, hidden_size=%d, depth=%d.", input_size, hidden_size, depth)
        logger.info("Using spectral clustering with %d (sample_points=%d).", self.n_clusters, self.clustering_sample_points)
        logger.info("")

        if input_projection:
            self.proj = nn.Sequential(Rearrange("B S F -> B (S F)"), nn.Linear(input_size, hidden_size))
        else:
            assert input_size == hidden_size, "Without a projection, input_size and hidden size must match"
            self.proj = nn.Sequential(Rearrange("B S F -> B (S F)"))

        # Build the encoder and decoder stages for HiERO
        self.up_stages = nn.ModuleList([self.build_stage(conv, n_layers, dropout) for _ in range(depth)])
        self.down_stages = nn.ModuleList([self.build_stage(conv, n_layers, dropout) for _ in range(depth)])

        # Initialize the KMeans model for HiERO clustering
        self.km = KMeans(n_clusters=n_clusters, seed=42, verbose=False)

    @property
    def depth(self) -> int:
        """Return the depth of the GNN hierarchy.

        Returns
        -------
        int
            The depth of the GNN hierarchy.
        """
        return len(self.up_stages)

    def build_stage(self, conv_config: dict, n_layers: int = 2, dropout: float = 0.0) -> nn.Module:
        """Build a stage in the GNN Unet.

        Parameters
        ----------
        conv_config : Dict[Any, Any]
            Configuration of the GNN layer.
        n_layers : int
            Number of layers in the stage, by default 2.
        dropout : float
            Dropout value, by default 0.0.

        Returns
        -------
        gnn.Sequential
            The GNN stage.
        """
        layers = []

        layers.append((nn.Dropout(dropout), "x -> x"))

        # 1. convolutional layer
        for _ in range(n_layers - 1):
            layers.append(self.build_conv_layer(conv_config))
            layers.append((nn.LeakyReLU(0.2), "x -> x"))

        layers.append(self.build_conv_layer(conv_config))

        # 2. normalization layer
        layers.append((nn.LayerNorm(conv_config["out_channels"]), "x -> x"))  # type: ignore

        # 3. activation layer
        layers.append((nn.LeakyReLU(0.2), "x -> x"))

        return gnn.Sequential("x, edge_index, pos, batch", layers)

    def build_conv_layer(self, config: Any) -> Tuple[nn.Module, str]:
        """Build a convolutional layer.

        Parameters
        ----------
        config : Dict[Any, Any]
            Configuration of the convolutional layer.

        Returns
        -------
        Tuple[nn.Module, str]
            A GNN layer and its signature (compatible with gnn.Sequential).
        """
        conv_layer = hydra.utils.instantiate(config)

        return (conv_layer, "x, edge_index, pos, batch -> x")

    def forward(self, data: Data, *args, **kwargs) -> Data:
        """Forward the graphs through the encoder/decoder architecture.

        Parameters
        ----------
        data : Data
            Input graphs to the encoder.

        Returns
        -------
        Data
            Output graphs from the decoder
        """

        # Keep only valid nodes
        x, batch, pos, indices = data.x, data.batch, data.pos, data.indices
        x, batch, pos, indices = x[data.mask], batch[data.mask], pos[data.mask], indices[data.mask]  # type: ignore

        # Compute the initial adjacency matrix of the graph
        edge_index = gnn.radius_graph(pos, self.k, batch, False)

        # Projection of the input graphs
        feat = self.proj(x)

        # Encoder step
        graph = Data(x=feat, edge_index=edge_index, pos=pos, batch=batch, indices=indices)
        last_graph, graphs = self.encoder(graph)

        # Decoder step
        graphs: List[Data] = self.decoder(last_graph, graphs)

        # For the moment we consider only the graph with highest resolution
        return Batch.from_data_list(graphs, follow_batch=["video"])

    def encoder(self, input_graph: Data) -> Tuple[Data, List[Data]]:
        """Forward the graph through the encoder.

        The Temporal Encoder implements local temporal reasoning, hierarchically aggregating
        information between temporally close segments.

        Parameters
        ----------
        input_graph : Data
            Input graph to the encoder.

        Returns
        -------
        Tuple[Data, List[Data]]
            Output of the encoder and list of intermediate outputs.
        """
        graphs = []

        feat, edge_index, pos, batch, indices = input_graph.x, input_graph.edge_index, input_graph.pos, input_graph.batch, input_graph.indices

        graphs = [Data(x=feat, edge_index=edge_index, pos=pos, video=batch, depth=torch.zeros_like(pos, dtype=torch.long), indices=indices)]

        for depth, stage in enumerate(self.up_stages):

            # Apply temporal pooling to the graph at this layer
            feat, pos, batch, indices = self.time_pooling(feat, pos, batch, indices)

            # Time-based edges and temporal convolution
            edge_index = gnn.radius_graph(pos / (2.0 ** (depth + 1)), self.k, batch, False)
            feat = feat + stage(feat, edge_index, pos, batch)

            last = Data(x=feat, edge_index=edge_index, pos=pos, video=batch, depth=(depth + 1) * torch.ones_like(pos, dtype=torch.long), indices=indices)
            if depth < (self.depth - 1):
                graphs.append(last)

        # Return the last output of the encoder and the intermediate outputs
        return last, graphs

    def decoder(self, last_graph: Data, graphs: List[Data]) -> List[Data]:
        """Decode the last graph into the intermediate graphs.

        Function-Aware Decoder extends temporal reasoning to nodes that may be temporally
        distant but functionally similar, by connecting nodes belonging to the same thread.

        Parameters
        ----------
        last_graph : Data
            Last output of the encoder.
        graphs : List[Data]
            Intermediate outputs of the encoder.

        Returns
        -------
        List[Data]
            Intermediate outputs of the decoder
        """
        output_graphs = []

        feat, pos, batch = last_graph.x, last_graph.pos, last_graph.video

        for i, (res, stage) in enumerate(zip(graphs[::-1], self.down_stages)):

            # Interpolate features back to original temporal resolution
            feat = res.x + knn_interpolate(feat, pos[:, None], res.pos[:, None], batch, res.video, k=2)
            edge_index, pos, batch, indices = res.edge_index, res.pos, res.video, res.indices

            depth = self.depth - i - 1  # from max_depth -> 0

            # Temporarily update the connectivity of the graph to connect segments of the video that are
            # functionally related to each other
            if depth > 0:

                # Update the connectivity of the graph...
                if self.training or self.clustering_at_inference:
                    updated_positions, cluster_assignments = self._clusterize(feat, pos / (2.0 ** (depth)), batch, indices)
                else:
                    updated_positions, cluster_assignments = indices.clone().float(), batch.clone()

                updated_edges = gnn.radius_graph(updated_positions, 1.5, cluster_assignments, False)

                res.assignments = cluster_assignments
                # and perform temporal reasoning on the updated graph
                feat = stage(feat, updated_edges, updated_positions, cluster_assignments)

            else:
                res.assignments = -1 * torch.ones_like(batch)
                feat = stage(feat, edge_index, pos, batch)

            res.x = feat

            output_graphs.append(res)

        # Intermediate outputs of the decoder
        return output_graphs[::-1]

    def time_pooling(self, x: Tensor, pos: Tensor, batch: Tensor, indices: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply temporal pooling on the input features.

        Parameters
        ----------
        x : Tensor
            Input features.
        pos : Tensor
            Input positions (temporal timestamps).
        batch : Tensor
            Input batch.
        indices : Tensor
            Input indices.

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor]
            Temporally pooled features, positions, batches, and indices.
        """

        if self.pool == "batch_subsampling":
            # Return one sample every two, regardless of video boundaries
            return x[::2], pos[::2], batch[::2], indices[::2]

        if self.pool in ["max", "mean"]:
            pooling_edges = gnn.radius_graph(indices.float(), 1.5, batch, True)

            # create a data object on the fly
            data = Data(x=x, edge_index=pooling_edges)
            data = gnn.pool.max_pool_neighbor_x(data) if self.pool == "max" else gnn.pool.avg_pool_neighbor_x(data)

            x = data.x  # type: ignore

        mask = indices % 2 == 0

        return x[mask], pos[mask], batch[mask], indices[mask] // 2

    @torch.no_grad()
    def _clusterize(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use spectral clustering to update the connectivity of the graph.

        This method uses spectral clustering of the features similarity matrix to find strongly connected regions
        of the video, which represent functionally related segments of the video.

        Since spectral clustering is a costly operation, we replace it with a fast approximation.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        pos : torch.Tensor
            Node positions (temporal timestamps).
        batch : torch.Tensor
            Batch tensor.
        indices : torch.Tensor
            Discrete time indices for the nodes.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated node positions and cluster assignments.
        """
        updated_positions = torch.zeros_like(pos)
        num_unique_videos = len(batch.unique())

        # Step 1: features interpolation
        # The graphs are subsampled to a fixed number of nodes, using interpolation to compute their node features.
        max_idx = gnn.pool.global_max_pool(indices, batch)
        normalized_indices = indices / max_idx[batch]
        # Uniformly sample num_sample_points points
        sample_points = torch.linspace(0, 1, self.clustering_sample_points, device=x.device).repeat(num_unique_videos)
        interp_batch = torch.arange(0, num_unique_videos, device=x.device).repeat_interleave(self.clustering_sample_points)
        interp_features = knn_interpolate(x, pos_x=normalized_indices[:, None], pos_y=sample_points[:, None], batch_x=batch, batch_y=interp_batch)
        batched_graphs = torch.stack(torch_geometric.utils.unbatch(interp_features, interp_batch))

        # Step 2: factorization of the graph laplacian
        # For each graph, we compute its graph laplacian and use K-Means to clusterize its top-k eigenvectors
        batched_graphs = torch.nn.functional.normalize(batched_graphs, p=2, dim=-1)
        W = torch.exp(torch.bmm(batched_graphs, batched_graphs.permute(0, 2, 1)) / 0.05)

        D_sqrt_inv = 1.0 / W.sum(-1).sqrt()
        D_sqrt_inv = torch.stack([torch.diag(video) for video in D_sqrt_inv.unbind(0)])

        L_norm = torch.eye(self.clustering_sample_points, device=W.device)[None] - torch.bmm(torch.bmm(D_sqrt_inv, W), D_sqrt_inv)

        # Batched eigendecomposition
        _, eigh = torch.linalg.eigh(L_norm)  # pylint: disable=not-callable
        topk_eigh = eigh[..., : self.n_clusters]

        # Step 3: compute the cluster assignments
        cluster_assignments = self.km.fit_predict(topk_eigh)
        cluster_assignments = rearrange(cluster_assignments, "b n -> (b n)", b=num_unique_videos, n=self.clustering_sample_points)
        cluster_assignments = knn_interpolate(cluster_assignments[:, None], pos_x=sample_points[:, None], pos_y=normalized_indices[:, None], batch_x=interp_batch, batch_y=batch, k=1).int().squeeze()
        cluster_assignments = batch * self.n_clusters + cluster_assignments

        for assignment in torch.unique(cluster_assignments):
            updated_positions[cluster_assignments == assignment] = 1.0 * torch.arange(0, (cluster_assignments == assignment).sum(), device=updated_positions.device).float()

        return updated_positions, cluster_assignments
