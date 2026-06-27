from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def cluster_connected_components(num_nodes: int, rows, cols, edge_scores):
    if len(rows) == 0:
        labels = np.arange(num_nodes, dtype=np.int64)
        return int(num_nodes), labels
    graph = coo_matrix((np.asarray(edge_scores, dtype=np.float32), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))), shape=(num_nodes, num_nodes)).tocsr()
    return connected_components(graph, directed=False, return_labels=True)


def relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    unique_labels = np.unique(labels)
    mapping = {int(label): index for index, label in enumerate(unique_labels.tolist())}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int64)


def build_cluster_edge_scores(labels: np.ndarray, rows, cols, edge_scores) -> Dict[int, List[float]]:
    cluster_scores = {int(label): [] for label in np.unique(labels).tolist()}
    for row, col, score in zip(rows, cols, edge_scores):
        if int(labels[int(row)]) == int(labels[int(col)]):
            cluster_scores[int(labels[int(row)])].append(float(score))
    return cluster_scores


def build_result_table(image_paths: List[str], labels: np.ndarray, view_stability: np.ndarray,
                       rows, cols, edge_scores, small_cluster_size: int) -> pd.DataFrame:
    labels = relabel_contiguous(labels)
    counts = pd.Series(labels).value_counts().to_dict()
    cluster_edge_scores = build_cluster_edge_scores(labels, rows, cols, edge_scores)
    records = []
    for sample_idx, (image_path, label, stability) in enumerate(zip(image_paths, labels.tolist(), view_stability.tolist())):
        cluster_size = int(counts[int(label)])
        if cluster_size == 1:
            cluster_type = 'singleton'
            cluster_conf = float(stability)
        elif cluster_size <= int(small_cluster_size):
            cluster_type = 'small'
            values = cluster_edge_scores.get(int(label), [])
            cluster_conf = float(np.mean(values)) if len(values) > 0 else None
        else:
            cluster_type = 'multi'
            values = cluster_edge_scores.get(int(label), [])
            cluster_conf = float(np.mean(values)) if len(values) > 0 else None
        records.append({
            'sample_idx': int(sample_idx),
            'img_path': image_path,
            'pseudo_label': int(label),
            'cluster_size': cluster_size,
            'cluster_type': cluster_type,
            'cluster_conf': cluster_conf,
            'is_singleton': bool(cluster_size == 1),
        })
    return pd.DataFrame(records)


def build_cluster_summary(result_df: pd.DataFrame, num_components: int) -> Dict[str, object]:
    cluster_sizes = result_df[['pseudo_label', 'cluster_size']].drop_duplicates()['cluster_size'].tolist()
    size_histogram = pd.Series(cluster_sizes).value_counts().sort_index().to_dict() if len(cluster_sizes) > 0 else {}
    return {
        'num_components': int(num_components),
        'num_singletons': int((result_df['cluster_type'] == 'singleton').sum()),
        'num_small_clusters': int(result_df.loc[result_df['cluster_type'] == 'small', 'pseudo_label'].nunique()),
        'num_multi_clusters': int(result_df.loc[result_df['cluster_type'] == 'multi', 'pseudo_label'].nunique()),
        'cluster_size_histogram': {str(int(key)): int(value) for key, value in size_histogram.items()},
    }
