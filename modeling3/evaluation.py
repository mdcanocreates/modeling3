"""
Evaluation utilities for Modeling 3.

Computes:
- Parent-match rate (% of synthetics that cluster with their parent)
- Intra-cluster distances
- Algorithm comparison metrics
"""

import numpy as np
from typing import Dict, List
from modeling3.clustering import compute_distance_matrix


def compute_parent_match_rate(
    labels: np.ndarray,
    parent_ids: List[str],
    sample_ids: List[str]
) -> Dict[str, float]:
    """
    Compute parent-match rate: % of generated images that cluster with their parent.
    
    Parameters
    ----------
    labels : np.ndarray
        Cluster labels (n_samples,)
    parent_ids : list
        List of parent IDs (e.g., ['CellA', 'CellA', 'CellB', ...])
    sample_ids : list
        List of sample IDs (e.g., ['CellA_original', 'CellA_alg1_001', ...])
    
    Returns
    -------
    dict
        Dictionary with:
        - 'overall_rate': Overall parent-match rate (0-1)
        - 'by_parent': Dict mapping parent_id to match rate
        - 'by_algorithm': Dict mapping algorithm to match rate
        - 'matches': List of (sample_id, parent_id, matched) tuples
    """
    if len(labels) != len(parent_ids) or len(labels) != len(sample_ids):
        raise ValueError("labels, parent_ids, and sample_ids must have same length")
    
    # Find original cell labels
    original_labels = {}
    for i, (sample_id, parent_id) in enumerate(zip(sample_ids, parent_ids)):
        if 'original' in sample_id.lower():
            original_labels[parent_id] = labels[i]
    
    # Check matches for generated images
    matches = []
    matches_by_parent = {pid: {'total': 0, 'matched': 0} for pid in set(parent_ids)}
    matches_by_algorithm = {'alg1': {'total': 0, 'matched': 0}, 'alg2': {'total': 0, 'matched': 0}}
    
    for i, (sample_id, parent_id) in enumerate(zip(sample_ids, parent_ids)):
        if 'original' in sample_id.lower():
            continue  # Skip originals
        
        # Determine algorithm
        if 'alg1' in sample_id.lower():
            algorithm = 'alg1'
        elif 'alg2' in sample_id.lower():
            algorithm = 'alg2'
        else:
            algorithm = 'unknown'
        
        # Check if matches parent's cluster
        parent_label = original_labels.get(parent_id)
        matched = (parent_label is not None) and (labels[i] == parent_label)
        
        matches.append((sample_id, parent_id, matched))
        
        # Update counts
        matches_by_parent[parent_id]['total'] += 1
        if matched:
            matches_by_parent[parent_id]['matched'] += 1
        
        if algorithm in matches_by_algorithm:
            matches_by_algorithm[algorithm]['total'] += 1
            if matched:
                matches_by_algorithm[algorithm]['matched'] += 1
    
    # Compute rates
    overall_matched = sum(1 for _, _, m in matches if m)
    overall_total = len(matches)
    overall_rate = overall_matched / overall_total if overall_total > 0 else 0.0
    
    by_parent_rates = {
        pid: (data['matched'] / data['total'] if data['total'] > 0 else 0.0)
        for pid, data in matches_by_parent.items()
    }
    
    by_algorithm_rates = {
        alg: (data['matched'] / data['total'] if data['total'] > 0 else 0.0)
        for alg, data in matches_by_algorithm.items()
    }
    
    return {
        'overall_rate': overall_rate,
        'by_parent': by_parent_rates,
        'by_algorithm': by_algorithm_rates,
        'matches': matches
    }


def compute_intra_cluster_distances(
    features: np.ndarray,
    labels: np.ndarray,
    metric: str = 'euclidean'
) -> Dict:
    """
    Compute intra-cluster distances for each cluster.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features)
    labels : np.ndarray
        Cluster labels (n_samples,)
    metric : str
        Distance metric
    
    Returns
    -------
    dict
        Dictionary with:
        - 'by_cluster': Dict mapping cluster_id to {'mean': float, 'std': float, 'max': float}
        - 'overall_mean': Overall mean intra-cluster distance
        - 'overall_std': Overall std intra-cluster distance
    """
    distance_matrix = compute_distance_matrix(features, metric=metric)
    
    unique_labels = np.unique(labels)
    by_cluster = {}
    
    all_distances = []
    
    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) < 2:
            # Single sample in cluster
            by_cluster[int(cluster_id)] = {
                'mean': 0.0,
                'std': 0.0,
                'max': 0.0,
                'n_samples': len(cluster_indices)
            }
            continue
        
        # Extract intra-cluster distances
        cluster_distances = []
        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                idx_i = cluster_indices[i]
                idx_j = cluster_indices[j]
                dist = distance_matrix[idx_i, idx_j]
                cluster_distances.append(dist)
        
        cluster_distances = np.array(cluster_distances)
        all_distances.extend(cluster_distances)
        
        by_cluster[int(cluster_id)] = {
            'mean': float(np.mean(cluster_distances)),
            'std': float(np.std(cluster_distances)),
            'max': float(np.max(cluster_distances)),
            'n_samples': len(cluster_indices)
        }
    
    all_distances = np.array(all_distances)
    
    return {
        'by_cluster': by_cluster,
        'overall_mean': float(np.mean(all_distances)) if len(all_distances) > 0 else 0.0,
        'overall_std': float(np.std(all_distances)) if len(all_distances) > 0 else 0.0
    }


def compare_algorithms(
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: List[str],
    parent_ids: List[str]
) -> Dict:
    """
    Compare Algorithm 1 vs Algorithm 2 in terms of realism/clustering.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix
    labels : np.ndarray
        Cluster labels
    sample_ids : list
        List of sample IDs
    parent_ids : list
        List of parent IDs
    
    Returns
    -------
    dict
        Dictionary with algorithm comparison metrics
    """
    # Separate by algorithm
    alg1_indices = [i for i, sid in enumerate(sample_ids) if 'alg1' in sid.lower()]
    alg2_indices = [i for i, sid in enumerate(sample_ids) if 'alg2' in sid.lower()]
    
    # Compute parent-match rates
    match_rates = compute_parent_match_rate(labels, parent_ids, sample_ids)
    
    # Compute intra-cluster distances for each algorithm
    alg1_distances = []
    alg2_distances = []
    
    for i in alg1_indices:
        cluster_id = labels[i]
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) > 1:
            # Compute mean distance to other samples in same cluster
            distances = []
            for j in cluster_indices:
                if i != j:
                    dist = np.linalg.norm(features[i] - features[j])
                    distances.append(dist)
            if distances:
                alg1_distances.append(np.mean(distances))
    
    for i in alg2_indices:
        cluster_id = labels[i]
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) > 1:
            distances = []
            for j in cluster_indices:
                if i != j:
                    dist = np.linalg.norm(features[i] - features[j])
                    distances.append(dist)
            if distances:
                alg2_distances.append(np.mean(distances))
    
    return {
        'alg1_match_rate': match_rates['by_algorithm'].get('alg1', 0.0),
        'alg2_match_rate': match_rates['by_algorithm'].get('alg2', 0.0),
        'alg1_mean_intra_distance': float(np.mean(alg1_distances)) if alg1_distances else 0.0,
        'alg2_mean_intra_distance': float(np.mean(alg2_distances)) if alg2_distances else 0.0,
        'alg1_n_samples': len(alg1_indices),
        'alg2_n_samples': len(alg2_indices)
    }

