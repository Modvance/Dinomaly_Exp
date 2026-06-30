from dataclasses import asdict, dataclass
import random
from typing import Optional, Tuple

import numpy as np


@dataclass
class AssignmentCalibration:
    tau_global: float
    tau_base: float
    tau_floor: float
    positive_mean: Optional[float]
    negative_mean: Optional[float]
    positive_std: Optional[float]
    negative_std: Optional[float]
    num_positive: int
    num_negative: int
    calibrated: bool

    def to_dict(self):
        return asdict(self)


class HistogramLikelihoodCalibrator:
    def __init__(self, num_bins: int = 100):
        self.num_bins = int(num_bins)
        self.bin_edges = None
        self.pos_density = None
        self.neg_density = None

    def fit(self, pos_sims, neg_sims):
        pos = np.asarray(pos_sims, dtype=np.float32)
        neg = np.asarray(neg_sims, dtype=np.float32)
        if pos.size == 0 or neg.size == 0:
            raise ValueError('both positive and negative similarities are required')
        lower = float(min(pos.min(), neg.min(), -1.0))
        upper = float(max(pos.max(), neg.max(), 1.0))
        if lower == upper:
            upper = lower + 1e-3
        self.bin_edges = np.linspace(lower, upper, self.num_bins + 1, dtype=np.float32)
        pos_hist, _ = np.histogram(pos, bins=self.bin_edges, density=False)
        neg_hist, _ = np.histogram(neg, bins=self.bin_edges, density=False)
        self.pos_density = (pos_hist.astype(np.float64) + 1.0) / (float(pos.size) + float(self.num_bins))
        self.neg_density = (neg_hist.astype(np.float64) + 1.0) / (float(neg.size) + float(self.num_bins))
        return self

    def _bin_index(self, sim: float) -> int:
        clipped = float(np.clip(sim, self.bin_edges[0], self.bin_edges[-1]))
        index = int(np.searchsorted(self.bin_edges, clipped, side='right') - 1)
        return int(np.clip(index, 0, len(self.pos_density) - 1))

    def predict_p_same(self, sim: float) -> float:
        if self.bin_edges is None or self.pos_density is None or self.neg_density is None:
            raise ValueError('calibrator is not fitted')
        index = self._bin_index(float(sim))
        likelihood_ratio = float(self.pos_density[index] / self.neg_density[index])
        return float(likelihood_ratio / (1.0 + likelihood_ratio))

    def predict_array(self, sims):
        return np.asarray([self.predict_p_same(float(sim)) for sim in np.asarray(sims).reshape(-1)], dtype=np.float32)


def build_positive_similarities(view_embeddings: np.ndarray) -> np.ndarray:
    if view_embeddings.ndim != 3:
        raise ValueError('view_embeddings must have shape [N, T, D]')
    num_views = int(view_embeddings.shape[1])
    if num_views <= 1:
        return np.ones((int(view_embeddings.shape[0]),), dtype=np.float32)
    scores = []
    for left_index in range(num_views):
        for right_index in range(left_index + 1, num_views):
            scores.append(np.sum(view_embeddings[:, left_index, :] * view_embeddings[:, right_index, :], axis=-1))
    return np.concatenate(scores, axis=0).astype(np.float32)


def build_negative_similarities(embeddings: np.ndarray, num_pairs: int, random_seed: int = 42) -> np.ndarray:
    num_samples = int(embeddings.shape[0])
    if num_samples <= 1:
        return np.zeros((0,), dtype=np.float32)
    rng = random.Random(int(random_seed))
    scores = []
    max_pairs = max(1, int(num_pairs))
    for _ in range(max_pairs):
        left_index, right_index = rng.sample(range(num_samples), 2)
        scores.append(float(np.dot(embeddings[left_index], embeddings[right_index])))
    return np.asarray(scores, dtype=np.float32)


def fit_similarity_calibrator(view_embeddings: np.ndarray, embeddings: np.ndarray, num_bins: int,
                              num_negative_pairs: int, random_seed: int = 42) -> Optional[HistogramLikelihoodCalibrator]:
    positive = build_positive_similarities(view_embeddings)
    negative = build_negative_similarities(embeddings, num_pairs=num_negative_pairs, random_seed=random_seed)
    if positive.size == 0 or negative.size == 0:
        return None
    return HistogramLikelihoodCalibrator(num_bins=num_bins).fit(positive, negative)


def _safe_mean(values: np.ndarray) -> Optional[float]:
    return None if values.size == 0 else float(np.mean(values))


def _safe_std(values: np.ndarray) -> Optional[float]:
    return None if values.size == 0 else float(np.std(values))


def _fallback_tau_global(positive: np.ndarray, negative: np.ndarray) -> float:
    if positive.size > 0 and negative.size > 0:
        positive_anchor = float(np.quantile(positive, 0.10))
        negative_anchor = float(np.quantile(negative, 0.99))
        return float(np.clip(0.5 * (positive_anchor + negative_anchor), -1.0, 1.0))
    if positive.size > 0:
        return float(np.clip(np.quantile(positive, 0.10), -1.0, 1.0))
    if negative.size > 0:
        return float(np.clip(np.quantile(negative, 0.99) + 0.05, -1.0, 1.0))
    return 0.60


def _derive_tau_global(positive: np.ndarray, negative: np.ndarray, num_bins: int) -> Tuple[float, bool]:
    if positive.size == 0 or negative.size == 0:
        return _fallback_tau_global(positive, negative), False
    calibrator = HistogramLikelihoodCalibrator(num_bins=num_bins).fit(positive, negative)
    centers = 0.5 * (calibrator.bin_edges[:-1] + calibrator.bin_edges[1:])
    probabilities = calibrator.predict_array(centers)
    eligible = centers[probabilities >= 0.5]
    if eligible.size > 0:
        return float(np.clip(eligible.min(), -1.0, 1.0)), True
    return _fallback_tau_global(positive, negative), True


def build_assignment_calibration(view_embeddings: np.ndarray, embeddings: np.ndarray, num_bins: int,
                                 num_negative_pairs: int, base_threshold_delta: float,
                                 assign_threshold_floor_offset: float, random_seed: int = 42) -> AssignmentCalibration:
    positive = build_positive_similarities(view_embeddings)
    negative = build_negative_similarities(embeddings, num_pairs=num_negative_pairs, random_seed=random_seed)
    tau_global, calibrated = _derive_tau_global(positive, negative, num_bins=int(num_bins))
    tau_base = float(np.clip(tau_global - float(base_threshold_delta), -1.0, 1.0))
    tau_floor = float(np.clip(tau_base - float(assign_threshold_floor_offset), -1.0, 1.0))
    return AssignmentCalibration(
        tau_global=float(tau_global),
        tau_base=float(tau_base),
        tau_floor=float(tau_floor),
        positive_mean=_safe_mean(positive),
        negative_mean=_safe_mean(negative),
        positive_std=_safe_std(positive),
        negative_std=_safe_std(negative),
        num_positive=int(positive.size),
        num_negative=int(negative.size),
        calibrated=bool(calibrated),
    )


def build_default_calibration(base_threshold_delta: float, assign_threshold_floor_offset: float) -> AssignmentCalibration:
    tau_global = 0.60
    tau_base = float(np.clip(tau_global - float(base_threshold_delta), -1.0, 1.0))
    tau_floor = float(np.clip(tau_base - float(assign_threshold_floor_offset), -1.0, 1.0))
    return AssignmentCalibration(
        tau_global=float(tau_global),
        tau_base=float(tau_base),
        tau_floor=float(tau_floor),
        positive_mean=None,
        negative_mean=None,
        positive_std=None,
        negative_std=None,
        num_positive=0,
        num_negative=0,
        calibrated=False,
    )
