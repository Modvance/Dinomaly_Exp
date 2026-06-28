from typing import Dict, List

import numpy as np
import torch


def l2_normalize(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return tensor / torch.clamp(torch.linalg.norm(tensor, dim=dim, keepdim=True), min=eps)


def aggregate_multiview(view_tensors: List[torch.Tensor], normalize: bool = True) -> torch.Tensor:
    stacked = torch.stack(view_tensors, dim=1)
    aggregated = stacked.mean(dim=1)
    if normalize:
        aggregated = l2_normalize(aggregated)
    return aggregated


def stack_view_embeddings(view_tensors: List[torch.Tensor], normalize: bool = True) -> torch.Tensor:
    stacked = torch.stack(view_tensors, dim=1)
    if normalize:
        stacked = l2_normalize(stacked, dim=-1)
    return stacked


def compute_view_stability(view_embeddings: List[torch.Tensor]) -> torch.Tensor:
    if len(view_embeddings) == 0:
        return torch.empty(0, dtype=torch.float32)
    if len(view_embeddings) == 1:
        return torch.ones(view_embeddings[0].shape[0], device=view_embeddings[0].device)
    normalized = [l2_normalize(embedding) for embedding in view_embeddings]
    scores = []
    for left_index in range(len(normalized)):
        for right_index in range(left_index + 1, len(normalized)):
            scores.append((normalized[left_index] * normalized[right_index]).sum(dim=-1))
    return torch.stack(scores, dim=0).mean(dim=0)


def build_feature_arrays(view_embeddings: List[torch.Tensor], normalize: bool = True) -> Dict[str, np.ndarray]:
    embedding = aggregate_multiview(view_embeddings, normalize=normalize).cpu().numpy().astype(np.float32)
    stacked_views = stack_view_embeddings(view_embeddings, normalize=normalize).cpu().numpy().astype(np.float32)
    stability = compute_view_stability(view_embeddings).cpu().numpy().astype(np.float32)
    return {
        'embeddings': embedding,
        'view_embeddings': stacked_views,
        'view_stability': stability,
    }
