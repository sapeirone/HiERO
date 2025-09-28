"""Clustering utils for Ego4d Goal-Step evaluation"""

from typing import List, Tuple

import numpy as np
import torch
from sklearn.cluster import SpectralClustering


def clusterize(features: torch.Tensor, n: int, temperature: float = 0.05) -> np.ndarray:
    """Compute cluster assignments using spectral clustering.

    Parameters
    ----------
    features : torch.Tensor
        Input features.
    n : int
        Number of clusters.
    temperature : float, optional
        Temperature parameter for similarity computation, by default 0.05

    Returns
    -------
    np.ndarray
        Cluster assignments.
    """
    norm_features = features / features.norm(p=2, dim=-1, keepdim=True)

    similarity_matrix = torch.exp((norm_features @ norm_features.T) / temperature).cpu().numpy()
    return SpectralClustering(n_clusters=n, affinity="precomputed").fit_predict(similarity_matrix)


def compress(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Compress the cluster assignments.

    This function takes a cluster assignments mask and computes a compressed version.

    Example:
        >>> compress(np.array([0, 0, 1, 1, 1, 2, 2]))
        [(0, 2), (2, 3), (5, 2)]

    Parameters
    ----------
    mask : np.ndarray
        Input cluster assignments mask.

    Returns
    -------
    List[Tuple[int, int]]
        Compressed representation as a list of (start, length) tuples.
    """
    # Initialize compressed representation
    compressed = []
    current_value = mask[0]
    count = 1

    # Iterate through the tensor
    for i in range(1, len(mask)):
        if mask[i] == current_value:
            count += 1
        else:
            compressed.append((i - count, count))
            current_value = mask[i]
            count = 1

    compressed.append((i - count, count))
    return compressed
