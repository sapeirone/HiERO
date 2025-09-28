import cv2
import torch
from torch.nn import functional as F

import numpy as np

from sklearn.cluster import SpectralClustering

def clusterize(features: torch.Tensor, n: int, temp=0.05):
    """
    Perform spectral clustering on the given features.
    Args:
        features (torch.Tensor): The input features to cluster, expected to be a 2D tensor.
        n (int): The number of clusters to form.
        temp (float, optional): Temperature parameter for scaling the affinity matrix. Default is 0.05.
    Returns:
        numpy.ndarray: An array of cluster labels for each feature.
    """

    features = F.normalize(features, p=2, dim=-1)
    affinity = torch.exp((features @ features.T) / temp).cpu().numpy()
    return SpectralClustering(n_clusters=n, affinity="precomputed").fit_predict(affinity)


def get_fps(video_path: str) -> float:
    """
    Get the frames per second of a video.
    Args:
        video_path (str): The path to the video.
    Returns:
        int: The frames per second of the video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps)

