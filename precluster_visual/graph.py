from typing import Dict, List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def adaptive_k(num_samples: int, k_min: int, k_max: int) -> int:
    if num_samples <= 1:
        return 1
    return int(min(k_max, max(k_min, int(np.sqrt(num_samples)))))


def build_mutual_knn_candidates(embeddings: np.ndarray, k: int, metric: str = 'cosine'):
    num_samples = embeddings.shape[0]
    effective_k = min(max(1, k), max(1, num_samples - 1))
    neighbors = NearestNeighbors(n_neighbors=effective_k + 1, metric=metric, algorithm='brute')
    neighbors.fit(embeddings)
    indices = neighbors.kneighbors(embeddings, return_distance=False)
    neighbor_lists = [list(map(int, row[1:])) for row in indices]
    neighbor_sets = [set(row) for row in neighbor_lists]
    candidate_pairs = []
    for src in range(num_samples):
        for dst in neighbor_lists[src]:
            if src == dst:
                continue
            if src in neighbor_sets[dst]:
                candidate_pairs.append((int(src), int(dst)))
    return effective_k, neighbor_lists, neighbor_sets, candidate_pairs


def _compute_similarity(embeddings: np.ndarray, src: int, dst: int) -> float:
    return float(np.dot(embeddings[src], embeddings[dst]))


def _compute_snn(left_neighbors: List[int], right_neighbors: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return float(len(set(left_neighbors).intersection(right_neighbors)) / max(1, k))


def score_candidate_edges(embeddings: np.ndarray, neighbor_lists: List[List[int]],
                          candidate_pairs: List[Tuple[int, int]], k: int, calibrator=None,
                          connect_if_p_same_gt: float = 0.5, use_snn: bool = True):
    edge_records = []
    for src, dst in candidate_pairs:
        similarity = _compute_similarity(embeddings, src, dst)
        p_same = None
        if calibrator is not None:
            p_same = float(calibrator.predict_p_same(similarity))
            if p_same <= float(connect_if_p_same_gt):
                continue
            base_weight = p_same
        else:
            base_weight = similarity
        snn = _compute_snn(neighbor_lists[src], neighbor_lists[dst], k=k) if use_snn else 1.0
        edge_weight = float(base_weight * (0.5 + 0.5 * snn)) if use_snn else float(base_weight)
        edge_records.append({
            'src': int(src),
            'dst': int(dst),
            'edge_score': edge_weight,
            'similarity': float(similarity),
            'p_same': None if p_same is None else float(p_same),
            'snn': float(snn),
        })
    return edge_records


def retain_top_edges_per_node(edge_records, max_edges_per_node: int):
    top_by_src: Dict[int, List[dict]] = {}
    for record in edge_records:
        top_by_src.setdefault(int(record['src']), []).append(record)
    retained_directed = []
    for src, records in top_by_src.items():
        for record in sorted(records, key=lambda item: (-item['edge_score'], item['dst']))[:int(max_edges_per_node)]:
            retained_directed.append(record)
    undirected = {}
    for record in retained_directed:
        key = tuple(sorted((int(record['src']), int(record['dst']))))
        current = undirected.get(key)
        if current is None or float(record['edge_score']) > float(current['edge_score']):
            undirected[key] = record
    retained_rows, retained_cols, retained_scores = [], [], []
    extras = {'similarity': [], 'snn': []}
    has_p_same = any(record.get('p_same') is not None for record in undirected.values())
    if has_p_same:
        extras['p_same'] = []
    for (src, dst), record in sorted(undirected.items()):
        retained_rows.extend([src, dst])
        retained_cols.extend([dst, src])
        retained_scores.extend([record['edge_score'], record['edge_score']])
        extras['similarity'].extend([record['similarity'], record['similarity']])
        extras['snn'].extend([record['snn'], record['snn']])
        if has_p_same:
            value = float(record['p_same']) if record.get('p_same') is not None else np.nan
            extras['p_same'].extend([value, value])
    return retained_rows, retained_cols, retained_scores, extras


def build_graph(feature_bundle: Dict[str, np.ndarray], config, calibrator=None):
    embeddings = feature_bundle['embeddings']
    num_samples = embeddings.shape[0]
    requested_k = adaptive_k(num_samples, int(config.knn.k_min), int(config.knn.k_max))
    effective_k, neighbor_lists, _, candidate_pairs = build_mutual_knn_candidates(
        embeddings,
        k=requested_k,
        metric=config.knn.metric,
    )
    edge_records = score_candidate_edges(
        embeddings,
        neighbor_lists,
        candidate_pairs,
        k=effective_k,
        calibrator=calibrator if bool(config.calibration.enabled) else None,
        connect_if_p_same_gt=float(config.calibration.connect_if_p_same_gt),
        use_snn=bool(config.knn.use_snn),
    )
    rows, cols, edge_scores, extras = retain_top_edges_per_node(edge_records, max_edges_per_node=int(config.graph.max_edges_per_node))
    return {
        'k': int(effective_k),
        'requested_k': int(requested_k),
        'candidate_edge_count': int(len(edge_records)),
        'retained_edge_count': int(len(rows) // 2),
        'rows': rows,
        'cols': cols,
        'edge_scores': edge_scores,
        'extras': extras,
    }
