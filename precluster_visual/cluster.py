from typing import Dict, List

import numpy as np
import pandas as pd


def relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    unique_labels = np.unique(labels)
    mapping = {int(label): index for index, label in enumerate(unique_labels.tolist())}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int64)


def build_result_table(image_paths: List[str], labels: np.ndarray, prototype_ids: np.ndarray,
                       view_stability: np.ndarray, assignment_scores: np.ndarray,
                       assignment_margins: np.ndarray, assignment_thresholds: np.ndarray,
                       prototype_confidences: np.ndarray, small_cluster_size: int) -> pd.DataFrame:
    labels = relabel_contiguous(np.asarray(labels, dtype=np.int64))
    counts = pd.Series(labels).value_counts().to_dict()
    records = []
    for sample_idx, image_path in enumerate(image_paths):
        label = int(labels[int(sample_idx)])
        cluster_size = int(counts[int(label)])
        if cluster_size == 1:
            cluster_type = 'singleton'
        elif cluster_size <= int(small_cluster_size):
            cluster_type = 'small'
        else:
            cluster_type = 'multi'
        records.append({
            'sample_idx': int(sample_idx),
            'img_path': image_path,
            'pseudo_label': int(label),
            'cluster_size': cluster_size,
            'cluster_type': cluster_type,
            'cluster_conf': float(prototype_confidences[int(sample_idx)]),
            'is_singleton': bool(cluster_size == 1),
            'assignment_score': float(assignment_scores[int(sample_idx)]),
            'assignment_margin': float(assignment_margins[int(sample_idx)]),
            'assignment_threshold': float(assignment_thresholds[int(sample_idx)]),
            'view_stability': float(view_stability[int(sample_idx)]),
            'prototype_id': int(prototype_ids[int(sample_idx)]),
        })
    return pd.DataFrame(records)


def build_cluster_summary(result_df: pd.DataFrame, num_clusters: int) -> Dict[str, object]:
    cluster_sizes = result_df[['pseudo_label', 'cluster_size']].drop_duplicates()['cluster_size'].tolist()
    size_histogram = pd.Series(cluster_sizes).value_counts().sort_index().to_dict() if len(cluster_sizes) > 0 else {}
    return {
        'num_clusters': int(num_clusters),
        'num_singletons': int((result_df['cluster_type'] == 'singleton').sum()),
        'num_small_clusters': int(result_df.loc[result_df['cluster_type'] == 'small', 'pseudo_label'].nunique()),
        'num_multi_clusters': int(result_df.loc[result_df['cluster_type'] == 'multi', 'pseudo_label'].nunique()),
        'cluster_size_histogram': {str(int(key)): int(value) for key, value in size_histogram.items()},
        'mean_assignment_score': float(result_df['assignment_score'].mean()) if len(result_df) > 0 else 0.0,
        'mean_assignment_margin': float(result_df['assignment_margin'].mean()) if len(result_df) > 0 else 0.0,
        'mean_view_stability': float(result_df['view_stability'].mean()) if len(result_df) > 0 else 0.0,
    }


def build_prototype_summary(prototype_ids: np.ndarray, prototype_thresholds: np.ndarray,
                            prototype_confidences: np.ndarray) -> Dict[str, object]:
    if len(prototype_ids) == 0:
        return {
            'num_prototypes': 0,
            'prototype_size_histogram': {},
            'mean_prototype_threshold': 0.0,
            'mean_prototype_confidence': 0.0,
        }
    frame = pd.DataFrame({
        'prototype_id': np.asarray(prototype_ids, dtype=np.int64),
        'prototype_threshold': np.asarray(prototype_thresholds, dtype=np.float32),
        'prototype_confidence': np.asarray(prototype_confidences, dtype=np.float32),
    })
    grouped = frame.groupby('prototype_id', sort=True)
    sizes = grouped.size().tolist()
    threshold_values = grouped['prototype_threshold'].first().tolist()
    confidence_values = grouped['prototype_confidence'].first().tolist()
    histogram = pd.Series(sizes).value_counts().sort_index().to_dict() if len(sizes) > 0 else {}
    return {
        'num_prototypes': int(len(sizes)),
        'prototype_size_histogram': {str(int(key)): int(value) for key, value in histogram.items()},
        'mean_prototype_threshold': float(np.mean(threshold_values)) if len(threshold_values) > 0 else 0.0,
        'mean_prototype_confidence': float(np.mean(confidence_values)) if len(confidence_values) > 0 else 0.0,
    }
