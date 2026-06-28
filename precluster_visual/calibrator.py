import random
from typing import Optional

import numpy as np


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
