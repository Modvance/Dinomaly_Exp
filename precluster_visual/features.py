from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def l2_normalize(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return tensor / torch.clamp(torch.linalg.norm(tensor, dim=dim, keepdim=True), min=eps)


def compute_foreground_weights(cls_tokens: torch.Tensor, patch_tokens: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    cls_tokens = l2_normalize(cls_tokens, dim=-1, eps=eps)
    patch_tokens = l2_normalize(patch_tokens, dim=-1, eps=eps)
    similarity = (patch_tokens * cls_tokens[:, None, None, :]).sum(dim=-1)
    min_value = similarity.amin(dim=(1, 2), keepdim=True)
    max_value = similarity.amax(dim=(1, 2), keepdim=True)
    denom = torch.clamp(max_value - min_value, min=eps)
    weights = (similarity - min_value) / denom
    uniform_mask = ((max_value - min_value) < eps).to(weights.dtype)
    uniform_weights = torch.ones_like(weights)
    return weights * (1.0 - uniform_mask) + uniform_weights * uniform_mask


def weighted_mean(features: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    numerator = (features * weights[..., None]).sum(dim=(1, 2))
    denominator = torch.clamp(weights.sum(dim=(1, 2), keepdim=False)[..., None], min=eps)
    return numerator / denominator


def compute_object_feature(patch_tokens: torch.Tensor, foreground_weights: torch.Tensor) -> torch.Tensor:
    return weighted_mean(patch_tokens, foreground_weights)


def compute_structure_feature(patch_tokens: torch.Tensor, foreground_weights: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
    gh, gw = int(grid[0]), int(grid[1])
    batch_size, height, width, dim = patch_tokens.shape
    row_edges = torch.linspace(0, height, gh + 1, device=patch_tokens.device).round().long()
    col_edges = torch.linspace(0, width, gw + 1, device=patch_tokens.device).round().long()
    pooled_regions = []
    for row in range(gh):
        for col in range(gw):
            r0, r1 = int(row_edges[row].item()), int(row_edges[row + 1].item())
            c0, c1 = int(col_edges[col].item()), int(col_edges[col + 1].item())
            region_tokens = patch_tokens[:, r0:r1, c0:c1, :]
            region_weights = foreground_weights[:, r0:r1, c0:c1]
            pooled_regions.append(weighted_mean(region_tokens, region_weights))
    return torch.cat(pooled_regions, dim=-1).reshape(batch_size, gh * gw * dim)


def compute_texture_feature(patch_tokens: torch.Tensor, foreground_weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = weighted_mean(patch_tokens, foreground_weights, eps=eps)
    centered = patch_tokens - mean[:, None, None, :]
    variance = weighted_mean(centered ** 2, foreground_weights, eps=eps)
    std = torch.sqrt(torch.clamp(variance, min=eps))
    return torch.cat([mean, std], dim=-1)


def fuse_features(object_feature: torch.Tensor, structure_feature: torch.Tensor, texture_feature: torch.Tensor, config) -> torch.Tensor:
    if config.features.normalize_each_feature:
        object_feature = l2_normalize(object_feature)
        structure_feature = l2_normalize(structure_feature)
        texture_feature = l2_normalize(texture_feature)
    fused = torch.cat([
        float(config.features.object_weight) * object_feature,
        float(config.features.structure_weight) * structure_feature,
        float(config.features.texture_weight) * texture_feature,
    ], dim=-1)
    if config.features.normalize_final_feature:
        fused = l2_normalize(fused)
    return fused


def aggregate_multiview(view_tensors: List[torch.Tensor], normalize: bool = True) -> torch.Tensor:
    stacked = torch.stack(view_tensors, dim=1)
    aggregated = stacked.mean(dim=1)
    if normalize:
        aggregated = l2_normalize(aggregated)
    return aggregated


def compute_view_stability(view_embeddings: List[torch.Tensor]) -> torch.Tensor:
    if len(view_embeddings) <= 1:
        return torch.ones(view_embeddings[0].shape[0], device=view_embeddings[0].device)
    normalized = [l2_normalize(embedding) for embedding in view_embeddings]
    scores = []
    for left_index in range(len(normalized)):
        for right_index in range(left_index + 1, len(normalized)):
            scores.append((normalized[left_index] * normalized[right_index]).sum(dim=-1))
    return torch.stack(scores, dim=0).mean(dim=0)


def build_feature_bundle(cls_tokens: torch.Tensor, patch_tokens_high: torch.Tensor, patch_tokens_mid: torch.Tensor, config) -> Dict[str, torch.Tensor]:
    foreground_weights = compute_foreground_weights(cls_tokens, patch_tokens_high, eps=float(config.features.foreground_eps))
    object_feature = compute_object_feature(patch_tokens_high, foreground_weights)
    structure_feature = compute_structure_feature(patch_tokens_high, foreground_weights, tuple(config.features.structure_grid))
    texture_feature = compute_texture_feature(patch_tokens_mid, foreground_weights, eps=float(config.features.foreground_eps))
    fused = fuse_features(object_feature, structure_feature, texture_feature, config)
    return {
        'foreground_weights': foreground_weights,
        'object_feature': object_feature,
        'structure_feature': structure_feature,
        'texture_feature': texture_feature,
        'embedding': fused,
    }
