import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity


EPS = 1e-12


def compute_self_sim(features: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError('tail sampler expects image-level features with shape [N, D]')
    features = features.to(torch.float32)
    if normalize:
        features = F.normalize(features, dim=-1)
    return features @ features.t()


def predict_class_sizes(self_sim: torch.Tensor, threshold: float, vote_type: str = 'mean') -> torch.Tensor:
    mask = self_sim >= float(threshold)
    counts = mask.sum(dim=1, keepdim=True).to(torch.float32)
    count_map = mask.to(torch.float32) * counts

    if vote_type == 'mean':
        nonzero = torch.count_nonzero(count_map, dim=0).clamp_min(1)
        class_sizes = count_map.sum(dim=0) / nonzero
        return class_sizes.to(torch.float32)

    if vote_type == 'nearest':
        nonzero = torch.count_nonzero(count_map, dim=0).clamp_min(1)
        columnwise_means = count_map.sum(dim=0, keepdim=True) / nonzero.unsqueeze(0)
        indices = torch.argmin((count_map - columnwise_means).abs(), dim=0)
        return count_map[indices, torch.arange(count_map.shape[1])].to(torch.float32)

    if vote_type == 'mode':
        values = []
        count_map_np = count_map.cpu().numpy()
        for column in range(count_map_np.shape[1]):
            column_values = count_map_np[:, column]
            column_values = column_values[column_values > 0]
            if column_values.size == 0:
                values.append(0.0)
                continue
            uniq, counts_np = np.unique(column_values, return_counts=True)
            values.append(float(uniq[np.argmax(counts_np)]))
        return torch.tensor(values, dtype=torch.float32)

    raise ValueError(f'unsupported vote_type: {vote_type}')


def predict_num_samples_per_class(class_sizes: torch.Tensor, round_class_sizes: bool = True) -> torch.Tensor:
    class_sizes = class_sizes.detach().to(torch.float32)
    if round_class_sizes:
        class_sizes = torch.round(class_sizes).to(torch.long)
    sorted_sizes = torch.sort(class_sizes, descending=False)[0]
    sorted_sizes = torch.maximum(sorted_sizes, torch.ones_like(sorted_sizes))

    num_samples_per_class = []
    while len(sorted_sizes) > 0:
        current = int(sorted_sizes[0].item())
        if len(sorted_sizes) < current:
            if len(num_samples_per_class) == 0:
                num_samples_per_class.append(len(sorted_sizes))
            else:
                num_samples_per_class[-1] += len(sorted_sizes)
            break

        current = int(torch.round(sorted_sizes[:current].float().mean()).item())
        current = min(current, len(sorted_sizes))
        num_samples_per_class.append(current)
        sorted_sizes = sorted_sizes[current:]

    if len(num_samples_per_class) == 0:
        return torch.zeros(0, dtype=torch.long)
    return torch.sort(torch.tensor(num_samples_per_class, dtype=torch.long), descending=True)[0]


def compute_orthogonal_distances(points: torch.Tensor, line_points: torch.Tensor) -> torch.Tensor:
    points_np = points.detach().cpu().numpy()
    line_np = line_points.detach().cpu().numpy()
    (x1, y1), (x2, y2) = line_np[0], line_np[1]
    dx, dy = x2 - x1, y2 - y1
    if dx == 0:
        return torch.tensor([abs(x - x1) for x, _ in points_np], dtype=torch.float32)

    slope = dy / dx
    intercept = y1 - slope * x1
    perp_slope = -1.0 / slope

    distances = []
    for x, y in points_np:
        perp_intercept = y - perp_slope * x
        x_inter = (perp_intercept - intercept) / (slope - perp_slope)
        y_inter = slope * x_inter + intercept
        distances.append(math.sqrt((x_inter - x) ** 2 + (y_inter - y) ** 2))
    return torch.tensor(distances, dtype=torch.float32)


def elbow(scores: torch.Tensor, sort: bool = True, quantize: bool = False) -> float:
    scores = scores.detach().to(torch.float32)
    scale = float(len(scores)) if quantize else 1.0
    if quantize:
        scores = (scores * scale).long().to(torch.float32)
    if sort:
        scores = torch.sort(scores, descending=True)[0]
    if scores.numel() == 1:
        return float(scores[0].item() / scale)

    points = torch.cat([torch.arange(len(scores), dtype=torch.float32)[:, None], scores[:, None]], dim=1)
    distances = compute_orthogonal_distances(points, points[[0, -1], :])
    idx = int(torch.argmax(distances).item())
    return float(scores[idx].item() / scale)


def _predict_max_k_max_within_percentile(num_samples_per_class: torch.Tensor, percentile: float = 0.15) -> int:
    sorted_counts = torch.sort(num_samples_per_class, descending=True)[0]
    budget = int(sorted_counts.sum().item() * percentile)
    cumulative = torch.flip(sorted_counts, dims=[0]).cumsum(dim=0)

    tail_classes = 0
    for value in cumulative:
        if int(value.item()) > budget:
            break
        tail_classes += 1

    index = min(len(sorted_counts) - tail_classes, len(sorted_counts) - 1)
    return int(sorted_counts[index].item())


def predict_max_k(num_samples_per_class: torch.Tensor, percentile: float = 0.15) -> int:
    if num_samples_per_class.numel() == 0:
        return 0
    max_k = int(elbow(num_samples_per_class.to(torch.float32)))
    if percentile < 1:
        max_k_percentile = _predict_max_k_max_within_percentile(num_samples_per_class, percentile=percentile)
        max_k = min(max_k, max_k_percentile)
    return max_k


def predict_few_shot_class_samples(class_sizes: torch.Tensor, percentile: float = 0.15) -> torch.Tensor:
    num_samples_per_class = predict_num_samples_per_class(class_sizes)
    max_k = predict_max_k(num_samples_per_class, percentile=percentile)
    return (class_sizes <= max_k).to(torch.long)


def compute_self_sim_min(self_sim: torch.Tensor, mode: str = 'min') -> torch.Tensor:
    minimums = self_sim.min(dim=1).values
    if mode == 'avg':
        return minimums.mean()
    if mode == 'min':
        return minimums.min()
    raise ValueError(f'unsupported min mode: {mode}')


def compute_sym_threshold(self_sim: torch.Tensor, mode: str = 'min') -> float:
    minimum = compute_self_sim_min(self_sim, mode=mode).clamp(-1 + EPS, 1 - EPS)
    return float(torch.cos(torch.acos(minimum) / 2.0).item())


def compute_threshold(
    self_sim: torch.Tensor,
    threshold_type: str = 'symmin',
    num_bootstrapping: int = 1,
    subsampling_ratio: float = 1.0,
) -> float:
    def _single(sim: torch.Tensor) -> float:
        if threshold_type == 'symmin':
            return compute_sym_threshold(sim, mode='min')
        if threshold_type == 'symavg':
            return compute_sym_threshold(sim, mode='avg')
        if threshold_type == 'indep':
            return float(np.cos(np.pi / 4.0))
        raise ValueError(f'unsupported threshold_type: {threshold_type}')

    if num_bootstrapping <= 1:
        return _single(self_sim)

    thresholds = []
    n = len(self_sim)
    batch = max(1, int(round(subsampling_ratio * n)))
    for _ in range(int(num_bootstrapping)):
        indices = torch.randint(0, n, (batch,))
        sim = self_sim.index_select(0, indices).index_select(1, indices)
        thresholds.append(_single(sim))
    return float(np.mean(thresholds))


def sample_few_shot(
    features: torch.Tensor,
    threshold_type: str = 'symmin',
    vote_type: str = 'mean',
    num_bootstrapping: int = 1,
    subsampling_ratio: float = 1.0,
    percentile: float = 0.15,
    return_class_sizes: bool = False,
):
    self_sim = compute_self_sim(features, normalize=True)
    threshold = compute_threshold(
        self_sim,
        threshold_type=threshold_type,
        num_bootstrapping=num_bootstrapping,
        subsampling_ratio=subsampling_ratio,
    )
    class_sizes = predict_class_sizes(self_sim, threshold=threshold, vote_type=vote_type)
    if return_class_sizes:
        return class_sizes

    is_few_shot = predict_few_shot_class_samples(class_sizes, percentile=percentile)
    sample_indices = torch.where(is_few_shot == 1)[0]
    return features[sample_indices], sample_indices


def _sort_scores(scores: np.ndarray, descending: bool = False, quantize: bool = True) -> np.ndarray:
    scores = np.asarray(scores)
    if quantize:
        scale = len(scores)
        scores = (scale * scores).astype(np.int64).astype(np.float64) / max(scale, 1)
    else:
        scores = scores.astype(np.float64)
    scores = np.sort(scores)
    if descending:
        scores = scores[::-1]
    return scores


def _double_criterion(criteria: np.ndarray) -> int:
    next_criteria = np.empty_like(criteria)
    for index in range(len(next_criteria) - 1):
        next_criteria[index] = np.max(criteria[index + 1:])
    next_criteria[-1] = criteria[-1]
    final = criteria / np.maximum(next_criteria, 1e-7)
    return int(np.argmax(final))


def scores_to_bin_counts(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    if iqr <= 0:
        return np.ones_like(scores, dtype=np.float64) * len(scores)
    bin_width = 2 * iqr / max(len(scores) ** (1 / 3), EPS)
    data_range = float(np.max(scores) - np.min(scores))
    if data_range <= 0 or bin_width <= 0:
        return np.ones_like(scores, dtype=np.float64) * len(scores)
    num_bins = max(1, int(np.round(data_range / bin_width)))
    hist, bin_edges = np.histogram(scores, bins=num_bins)
    counts = np.zeros_like(scores, dtype=np.float64)
    for index in range(num_bins):
        if index < num_bins - 1:
            in_bin = (scores >= bin_edges[index]) & (scores < bin_edges[index + 1])
        else:
            in_bin = scores >= bin_edges[index]
        counts[in_bin] = hist[index]
    return counts


def scores_to_kde(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    kde = KernelDensity(kernel='tophat', bandwidth='silverman').fit(values)
    return kde.score_samples(values)


def _compute_threshold_max_step(scores: np.ndarray) -> float:
    scores_sorted = _sort_scores(scores, descending=False)
    diffs = np.diff(scores_sorted)
    idx = int(np.argmax(diffs))
    return float((scores_sorted[idx + 1] + scores_sorted[idx]) / 2.0)


def _compute_threshold_double_max_step(scores: np.ndarray) -> float:
    scores_sorted = _sort_scores(scores, descending=False)
    diffs = np.diff(scores_sorted)
    idx = _double_criterion(diffs)
    return float((scores_sorted[idx] + scores_sorted[idx + 1]) / 2.0)


def _compute_threshold_max_step_min_num_neighbors(scores: np.ndarray) -> float:
    scores_sorted = _sort_scores(scores, descending=True)
    diffs = -np.diff(scores_sorted)
    crit1 = diffs
    crit2 = -np.log(np.arange(len(diffs)) + 1) + math.log(len(diffs))
    crit = crit1 * crit2
    idx = int(np.argmax(crit))
    return float((scores_sorted[idx] + scores_sorted[idx + 1]) / 2.0)


def _compute_threshold_double_min_bin_count(scores: np.ndarray) -> float:
    scores_sorted = _sort_scores(scores, descending=False)
    bin_counts = scores_to_bin_counts(scores_sorted)[:-1]
    idx = _double_criterion(bin_counts.max() - bin_counts)
    return float((scores_sorted[idx] + scores_sorted[idx + 1]) / 2.0)


def _compute_threshold_min_kde(scores: np.ndarray) -> float:
    scores_sorted = _sort_scores(scores, descending=False)
    kde_log_density = scores_to_kde(scores_sorted)[:-1]
    idx = int(np.argmin(kde_log_density))
    return float((scores_sorted[idx] + scores_sorted[idx + 1]) / 2.0)


def _compute_threshold_double_min_kde(scores: np.ndarray) -> float:
    scores_sorted = _sort_scores(scores, descending=False)
    kde_log_density = scores_to_kde(scores_sorted)[:-1]
    idx = _double_criterion(kde_log_density.max() - kde_log_density)
    return float((scores_sorted[idx] + scores_sorted[idx + 1]) / 2.0)


def _compute_threshold_half_min(scores: np.ndarray) -> float:
    minimum = np.clip(np.min(scores), -1 + EPS, 1 - EPS)
    return float(np.cos(np.arccos(minimum) / 2.0))


def _compute_threshold_trim_min(scores: np.ndarray) -> float:
    scores_sorted = _sort_scores(scores, descending=False)
    minimum = np.clip(scores_sorted.min(), -1 + EPS, 1 - EPS)
    threshold_half = np.cos(np.arccos(minimum) / 2.0)
    scores_half = scores_sorted[scores_sorted > threshold_half]
    if scores_half.size == 0:
        return float(threshold_half)
    keep = max(1, int(len(scores_half) * 0.85))
    return float(scores_half[-keep])


def _compute_threshold_truncate_min(scores: np.ndarray) -> float:
    scores_sorted = _sort_scores(scores, descending=False)
    minimum = np.clip(scores_sorted.min(), -1 + EPS, 1 - EPS)
    threshold_half = np.cos(np.arccos(minimum) / 2.0)
    scores_half = scores_sorted[scores_sorted > threshold_half]
    if scores_half.size == 0:
        return float(threshold_half)
    keep = max(1, int(len(scores_half) * 0.5))
    return float(scores_half[-keep])


def _compute_threshold_ruled_max_step(scores: np.ndarray) -> float:
    minimum = np.clip(np.min(scores), -1 + EPS, 1 - EPS)
    threshold_quarter = np.cos(np.arccos(minimum) / 8.0)
    threshold_half = np.cos(np.arccos(minimum) / 2.0)
    threshold_double = _compute_threshold_double_max_step(scores)
    if threshold_double >= threshold_quarter:
        return float(threshold_half)
    return float(threshold_double)


def compute_adaptive_thresholds(self_sim: torch.Tensor, threshold_type: str) -> torch.Tensor:
    self_sim_np = self_sim.detach().cpu().numpy()
    threshold_fns = {
        'max_step': _compute_threshold_max_step,
        'double_max_step': _compute_threshold_double_max_step,
        'max_step_min_num_neighbors': _compute_threshold_max_step_min_num_neighbors,
        'double_min_bin_count': _compute_threshold_double_min_bin_count,
        'min_kde': _compute_threshold_min_kde,
        'double_min_kde': _compute_threshold_double_min_kde,
        'half_min': _compute_threshold_half_min,
        'ruled_max_step': _compute_threshold_ruled_max_step,
        'trim_min': _compute_threshold_trim_min,
        'truncate_min': _compute_threshold_truncate_min,
    }
    if threshold_type not in threshold_fns:
        raise ValueError(f'unsupported adaptive threshold_type: {threshold_type}')
    fn = threshold_fns[threshold_type]
    thresholds = [fn(row) for row in self_sim_np]
    return torch.tensor(thresholds, dtype=torch.float32)[:, None]


def predict_adaptive_class_sizes(self_sim: torch.Tensor, threshold_type: str = 'double_max_step', vote_type: str = 'none') -> torch.Tensor:
    thresholds = compute_adaptive_thresholds(self_sim, threshold_type)
    mask = self_sim >= thresholds
    count_map = mask.to(torch.float32) * mask.sum(dim=1, keepdim=True).to(torch.float32)

    if vote_type == 'none':
        return count_map.mean(dim=1).to(torch.float32)
    if vote_type == 'mean':
        nonzero = torch.count_nonzero(count_map, dim=0).clamp_min(1)
        return (count_map.sum(dim=0) / nonzero).to(torch.float32)
    if vote_type == 'mode':
        values = []
        count_map_np = count_map.cpu().numpy()
        for column in range(count_map_np.shape[1]):
            column_values = count_map_np[:, column]
            column_values = column_values[column_values > 0]
            if column_values.size == 0:
                values.append(0.0)
                continue
            uniq, counts_np = np.unique(column_values, return_counts=True)
            values.append(float(uniq[np.argmax(counts_np)]))
        return torch.tensor(values, dtype=torch.float32)
    raise ValueError(f'unsupported adaptive vote_type: {vote_type}')


def adaptively_sample_few_shot(
    features: torch.Tensor,
    threshold_type: str = 'double_max_step',
    vote_type: str = 'none',
    percentile: float = 0.15,
    return_class_sizes: bool = False,
):
    self_sim = compute_self_sim(features, normalize=True)
    class_sizes = predict_adaptive_class_sizes(self_sim, threshold_type=threshold_type, vote_type=vote_type).squeeze()
    if return_class_sizes:
        return class_sizes

    is_few_shot = predict_few_shot_class_samples(class_sizes, percentile=percentile)
    sample_indices = torch.where(is_few_shot == 1)[0]
    return features[sample_indices], sample_indices


class TailSampler:
    def __init__(
        self,
        threshold_type: str = 'symmin',
        vote_type: str = 'mean',
        percentile: float = 0.15,
        num_bootstrapping: int = 1,
        subsampling_ratio: float = 1.0,
    ):
        self.threshold_type = threshold_type
        self.vote_type = vote_type
        self.percentile = float(percentile)
        self.num_bootstrapping = int(num_bootstrapping)
        self.subsampling_ratio = float(subsampling_ratio)

    def run(self, features: torch.Tensor, return_class_sizes: bool = False):
        tail_samples, tail_indices = sample_few_shot(
            features,
            threshold_type=self.threshold_type,
            vote_type=self.vote_type,
            num_bootstrapping=self.num_bootstrapping,
            subsampling_ratio=self.subsampling_ratio,
            percentile=self.percentile,
            return_class_sizes=False,
        )
        if return_class_sizes:
            class_sizes = sample_few_shot(
                features,
                threshold_type=self.threshold_type,
                vote_type=self.vote_type,
                num_bootstrapping=self.num_bootstrapping,
                subsampling_ratio=self.subsampling_ratio,
                percentile=self.percentile,
                return_class_sizes=True,
            )
            return tail_samples, tail_indices, class_sizes
        return tail_samples, tail_indices


class AdaptiveTailSampler:
    def __init__(
        self,
        threshold_type: str = 'double_max_step',
        vote_type: str = 'none',
        percentile: float = 0.15,
    ):
        self.threshold_type = threshold_type
        self.vote_type = vote_type
        self.percentile = float(percentile)

    def run(self, features: torch.Tensor, return_class_sizes: bool = False):
        tail_samples, tail_indices = adaptively_sample_few_shot(
            features,
            threshold_type=self.threshold_type,
            vote_type=self.vote_type,
            percentile=self.percentile,
            return_class_sizes=False,
        )
        if return_class_sizes:
            class_sizes = adaptively_sample_few_shot(
                features,
                threshold_type=self.threshold_type,
                vote_type=self.vote_type,
                percentile=self.percentile,
                return_class_sizes=True,
            )
            return tail_samples, tail_indices, class_sizes
        return tail_samples, tail_indices


def build_tail_sampler(
    sampler_type: str = 'adaptive_trim_mode',
    threshold_type: Optional[str] = None,
    vote_type: Optional[str] = None,
    percentile: float = 0.15,
):
    if sampler_type == 'tail':
        return TailSampler(
            threshold_type='symmin' if threshold_type is None else threshold_type,
            vote_type='mean' if vote_type is None else vote_type,
            percentile=percentile,
        ), 'TailSampler'

    if sampler_type == 'adaptive':
        return AdaptiveTailSampler(
            threshold_type='double_max_step' if threshold_type is None else threshold_type,
            vote_type='none' if vote_type is None else vote_type,
            percentile=percentile,
        ), 'AdaptiveTailSampler'

    if sampler_type == 'adaptive_trim_mode':
        return AdaptiveTailSampler(
            threshold_type='trim_min' if threshold_type is None else threshold_type,
            vote_type='mode' if vote_type is None else vote_type,
            percentile=percentile,
        ), 'AdaptiveTailSampler(trim_min,mode)'

    raise ValueError(f'unsupported sampler_type: {sampler_type}')
