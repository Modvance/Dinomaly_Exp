from typing import Dict, List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def adaptive_k(num_samples: int, k_min: int, k_max: int) -> int:
    if num_samples <= 1:
        return 1
    return int(min(k_max, max(k_min, int(np.sqrt(num_samples)))))


def _cosine_similarity(left: np.ndarray, right: np.ndarray, eps: float = 1e-8) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm < eps or right_norm < eps:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def build_mutual_knn_candidates(embeddings: np.ndarray, k: int, metric: str = 'cosine'):
    num_samples = embeddings.shape[0]
    effective_k = min(max(1, k), max(1, num_samples - 1))
    neighbors = NearestNeighbors(n_neighbors=effective_k + 1, metric=metric)
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


def score_candidate_edges(object_features: np.ndarray, structure_features: np.ndarray, texture_features: np.ndarray,
                          neighbor_lists: List[List[int]], candidate_pairs: List[Tuple[int, int]], k: int):
    edge_records = []
    for src, dst in candidate_pairs:
        s_obj = _cosine_similarity(object_features[src], object_features[dst])
        s_struct = _cosine_similarity(structure_features[src], structure_features[dst])
        s_tex = _cosine_similarity(texture_features[src], texture_features[dst])
        snn = float(len(set(neighbor_lists[src]).intersection(neighbor_lists[dst])) / max(1, k))
        score = float(min(s_obj, s_struct) * (0.5 + 0.5 * s_tex) * snn)
        edge_records.append({
            'src': int(src),
            'dst': int(dst),
            'edge_score': score,
            's_obj': float(s_obj),
            's_struct': float(s_struct),
            's_tex': float(s_tex),
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
    extras = {'s_obj': [], 's_struct': [], 's_tex': [], 'snn': []}
    for (src, dst), record in sorted(undirected.items()):
        retained_rows.extend([src, dst])
        retained_cols.extend([dst, src])
        retained_scores.extend([record['edge_score'], record['edge_score']])
        for key in extras.keys():
            extras[key].extend([record[key], record[key]])
    return retained_rows, retained_cols, retained_scores, extras


def build_graph(feature_bundle: Dict[str, np.ndarray], config):
    embeddings = feature_bundle['embedding']
    num_samples = embeddings.shape[0]
    requested_k = adaptive_k(num_samples, int(config.knn.k_min), int(config.knn.k_max))
    effective_k, neighbor_lists, _, candidate_pairs = build_mutual_knn_candidates(embeddings, k=requested_k, metric=config.knn.metric)
    edge_records = score_candidate_edges(
        feature_bundle['object_feature'],
        feature_bundle['structure_feature'],
        feature_bundle['texture_feature'],
        neighbor_lists,
        candidate_pairs,
        k=effective_k,
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
