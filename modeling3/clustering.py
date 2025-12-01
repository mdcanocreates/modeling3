"""
Clustering utilities for Modeling 3.

Implements:
- K-Means clustering (k=3)
- Hierarchical clustering (Ward linkage)
- Silhouette score computation
- Distance matrix computation
"""

import numpy as np
from typing import Dict, List
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import modeling3.config as config


def perform_kmeans_clustering(
    features: np.ndarray,
    k: int = None,
    random_state: int = None
) -> Dict:
    """
    Perform K-Means clustering on feature matrix.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features)
    k : int, optional
        Number of clusters. Defaults to config.N_CLUSTERS.
    random_state : int, optional
        Random seed. Defaults to config.RANDOM_SEED.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'labels': Cluster labels (n_samples,)
        - 'centers': Cluster centers (k, n_features)
        - 'inertia': Within-cluster sum of squares
        - 'n_clusters': Number of clusters
    """
    if k is None:
        k = config.N_CLUSTERS
    
    if random_state is None:
        random_state = config.RANDOM_SEED
    
    # Perform K-Means
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10
    )
    labels = kmeans.fit_predict(features)
    
    return {
        'labels': labels,
        'centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_,
        'n_clusters': k
    }


def perform_hierarchical_clustering(
    features: np.ndarray,
    method: str = 'ward',
    n_clusters: int = None
) -> Dict:
    """
    Perform hierarchical clustering using Ward linkage.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features)
    method : str
        Linkage method ('ward', 'complete', 'average', 'single')
    n_clusters : int, optional
        Number of clusters to extract. Defaults to config.N_CLUSTERS.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'labels': Cluster labels (n_samples,)
        - 'linkage_matrix': Linkage matrix for dendrogram
        - 'n_clusters': Number of clusters
    """
    if n_clusters is None:
        n_clusters = config.N_CLUSTERS
    
    # Compute linkage matrix
    linkage_matrix = linkage(features, method=method)
    
    # Perform clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=method
    )
    labels = clustering.fit_predict(features)
    
    return {
        'labels': labels,
        'linkage_matrix': linkage_matrix,
        'n_clusters': n_clusters
    }


def compute_silhouette_score(
    features: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute silhouette score for clustering.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features)
    labels : np.ndarray
        Cluster labels (n_samples,)
    
    Returns
    -------
    float
        Silhouette score (-1 to 1, higher is better)
    """
    if len(np.unique(labels)) < 2:
        # Need at least 2 clusters for silhouette score
        return -1.0
    
    score = silhouette_score(features, labels)
    return float(score)


def compute_distance_matrix(
    features: np.ndarray,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Compute pairwise distance matrix.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features)
    metric : str
        Distance metric ('euclidean', 'manhattan', etc.)
    
    Returns
    -------
    np.ndarray
        Distance matrix (n_samples, n_samples)
    """
    distances = pdist(features, metric=metric)
    distance_matrix = squareform(distances)
    return distance_matrix

