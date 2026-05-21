"""Comprehensive readout metrics operating on collections of trajectories."""

from typing import Any
import numpy as np
from musiq.common.schemas import ModelSpec, Trajectory
from musiq.schemas.results import MetricSeries

def compute_iq_clouds(trajectories: list[Trajectory], model_spec: ModelSpec) -> dict[str, list[tuple[float, float]]]:
    """Aggregate IQ clouds for all states across all trajectories."""
    clouds: dict[str, list[tuple[float, float]]] = {}
    
    for traj in trajectories:
        # Readout info is typically in traj.measurements or traj.classical
        # Assuming a standard readout record structure: {state_label: (I, Q)}
        readout = getattr(traj, "readout", {}) or getattr(traj, "classical", {}).get("readout", {})
        if not readout:
            continue
        
        # Readout might be a single value or a list of samples
        # If it's a dict {label: (I, Q)}, add it
        for label, val in readout.items():
            if isinstance(val, (tuple, list)) and len(val) == 2:
                clouds.setdefault(label, []).append((float(val[0]), float(val[1])))
    
    return clouds

def compute_centroids(iq_clouds: dict[str, list[tuple[float, float]]]) -> dict[str, tuple[float, float]]:
    """Compute the centroid (mean I, Q) for each state cloud."""
    centroids: dict[str, tuple[float, float]] = {}
    for label, points in iq_clouds.items():
        if not points:
            centroids[label] = (0.0, 0.0)
            continue
        pts = np.array(points)
        mean_i, mean_q = np.mean(pts, axis=0)
        centroids[label] = (float(mean_i), float(mean_q))
    return centroids

def compute_confusion_matrix(trajectories: list[Trajectory], centroids: dict[str, tuple[float, float]]) -> dict[str, dict[str, float]]:
    """Compute the state confusion matrix based on nearest-centroid assignment."""
    if not centroids:
        return {}
    
    labels = sorted(centroids.keys())
    matrix: dict[str, dict[str, float]] = {l1: {l2: 0.0 for l2 in labels} for l1 in labels}
    counts: dict[str, int] = {l: 0 for l in labels}

    for traj in trajectories:
        # True state from trajectory metadata or final population
        true_state = getattr(traj, "true_state", None) 
        if true_state not in labels:
            # Fallback: check which state has max population at end
            # This is a simplification; in real data, true_state is usually provided
            continue
        
        # Assigned state based on nearest centroid
        readout = getattr(traj, "readout", {}) or getattr(traj, "classical", {}).get("readout", {})
        if not readout: continue
        
        # Assume readout is a single (I, Q) pair for the measured state
        # If it's a dict, we take the measured one. For simplicity, we look for the observed IQ.
        # In a real system, the 'observed' IQ is a single point.
        observed_iq = list(readout.values())[0] if isinstance(readout, dict) else readout
        if not isinstance(observed_iq, (tuple, list)) or len(observed_iq) != 2:
            continue
        
        # Find nearest centroid
        obs = np.array(observed_iq)
        best_label = min(labels, key=lambda l: np.linalg.norm(obs - np.array(centroids[l])))
        
        matrix[true_state][best_label] += 1.0
        counts[true_state] += 1

    # Normalize
    for l1 in labels:
        if counts[l1] > 0:
            for l2 in labels:
                matrix[l1][l2] /= counts[l1]
                
    return matrix

def compute_readout_fidelity(confusion_matrix: dict[str, dict[str, float]]) -> float:
    """Compute overall readout fidelity as the average of diagonal elements."""
    labels = list(confusion_matrix.keys())
    if not labels: return 0.0
    
    sum_diag = sum(confusion_matrix[l][l] for l in labels)
    return float(sum_diag / len(labels))

def compute_snr(iq_clouds: dict[str, list[tuple[float, float]]], centroids: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Compute SNR as the distance between centroids divided by the average spread."""
    if len(centroids) < 2: return {}
    
    snrs: dict[str, float] = {}
    labels = list(centroids.keys())
    
    for l in labels:
        pts = np.array(iq_clouds.get(l, []))
        if len(pts) < 2:
            snrs[l] = 0.0
            continue
        
        # Spread as standard deviation of distance to centroid
        centroid = np.array(centroids[l])
        dists = np.linalg.norm(pts - centroid, axis=1)
        sigma = np.std(dists)
        
        # Distance to nearest other centroid
        min_dist = min(np.linalg.norm(centroid - np.array(centroids[other])) for other in labels if other != l)
        
        snrs[l] = float(min_dist / (2 * sigma)) if sigma > 0 else 0.0
        
    return snrs

# Registry-compatible wrappers for the hub
def metric_iq_clouds(trajectories: list[Trajectory], model_spec: ModelSpec, cfg: Any, ctx: Any) -> dict[str, Any]:
    res = compute_iq_clouds(trajectories, model_spec)
    return {"payload": res, "observable_updates": {}}

def metric_centroids(trajectories: list[Trajectory], model_spec: ModelSpec, cfg: Any, ctx: Any) -> dict[str, Any]:
    clouds = compute_iq_clouds(trajectories, model_spec)
    res = compute_centroids(clouds)
    return {"payload": res, "observable_updates": {}}

def metric_confusion_matrix(trajectories: list[Trajectory], model_spec: ModelSpec, cfg: Any, ctx: Any) -> dict[str, Any]:
    clouds = compute_iq_clouds(trajectories, model_spec)
    centroids = compute_centroids(clouds)
    res = compute_confusion_matrix(trajectories, centroids)
    return {"payload": res, "observable_updates": {}}

def metric_readout_fidelity(trajectories: list[Trajectory], model_spec: ModelSpec, cfg: Any, ctx: Any) -> dict[str, Any]:
    clouds = compute_iq_clouds(trajectories, model_spec)
    centroids = compute_centroids(clouds)
    matrix = compute_confusion_matrix(trajectories, centroids)
    res = compute_readout_fidelity(matrix)
    return {"payload": MetricSeries(values=[res]), "observable_updates": {"readout_fidelity": res}}

def metric_snr(trajectories: list[Trajectory], model_spec: ModelSpec, cfg: Any, ctx: Any) -> dict[str, Any]:
    clouds = compute_iq_clouds(trajectories, model_spec)
    centroids = compute_centroids(clouds)
    res = compute_snr(clouds, centroids)
    return {"payload": res, "observable_updates": {}}

__all__ = [
    "metric_iq_clouds", 
    "metric_centroids", 
    "metric_confusion_matrix", 
    "metric_readout_fidelity", 
    "metric_snr"
]