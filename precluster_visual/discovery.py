from typing import Dict, List

import numpy as np

from .prototypes import (
    assign_to_best_prototype,
    create_prototype,
    export_prototypes,
    merge_prototypes,
    refresh_prototype,
    update_prototype,
)


def discover_semantic_prototypes(embeddings: np.ndarray, view_stability: np.ndarray, calibration, config) -> Dict[str, np.ndarray]:
    num_samples = int(embeddings.shape[0])
    ordered_indices = np.argsort(-np.asarray(view_stability, dtype=np.float32), kind='stable').astype(np.int64)

    labels_before_merge = np.full((num_samples,), -1, dtype=np.int64)
    prototype_ids_before_merge = np.full((num_samples,), -1, dtype=np.int64)
    assignment_scores = np.zeros((num_samples,), dtype=np.float32)
    assignment_margins = np.zeros((num_samples,), dtype=np.float32)
    assignment_thresholds = np.zeros((num_samples,), dtype=np.float32)
    center_scores = np.full((num_samples,), np.nan, dtype=np.float32)
    support_scores = np.full((num_samples,), np.nan, dtype=np.float32)
    created_new_mask = np.zeros((num_samples,), dtype=np.bool_)

    prototypes = []
    next_prototype_id = 0

    for sorted_rank, sample_index in enumerate(ordered_indices.tolist()):
        embedding = embeddings[int(sample_index)]
        reliability = float(view_stability[int(sample_index)])
        decision = assign_to_best_prototype(
            sample_index=int(sample_index),
            embedding=embedding,
            reliability=reliability,
            prototypes=prototypes,
            embeddings=embeddings,
            config=config,
            base_threshold=float(calibration.tau_base),
            threshold_floor=float(calibration.tau_floor),
        )

        if decision.assigned and decision.prototype_id is not None:
            prototype = next(prototype for prototype in prototypes if int(prototype.id) == int(decision.prototype_id))
            update_prototype(
                prototype=prototype,
                sample_index=int(sample_index),
                embedding=embedding,
                reliability=reliability,
                assignment_score=float(decision.assignment_score),
                assignment_margin=float(decision.assignment_margin),
                config=config,
                base_threshold=float(calibration.tau_base),
                threshold_floor=float(calibration.tau_floor),
                embeddings=embeddings,
            )
        else:
            prototype = create_prototype(
                prototype_id=int(next_prototype_id),
                sample_index=int(sample_index),
                embeddings=embeddings,
                reliability=reliability,
                base_threshold=float(calibration.tau_base),
                config=config,
            )
            prototypes.append(prototype)
            next_prototype_id += 1

        labels_before_merge[int(sample_index)] = int(prototype.id)
        prototype_ids_before_merge[int(sample_index)] = int(prototype.id)
        assignment_scores[int(sample_index)] = float(decision.assignment_score)
        assignment_margins[int(sample_index)] = float(decision.assignment_margin)
        assignment_thresholds[int(sample_index)] = float(decision.assignment_threshold)
        if decision.center_score is not None:
            center_scores[int(sample_index)] = float(decision.center_score)
        if decision.support_score is not None:
            support_scores[int(sample_index)] = float(decision.support_score)
        created_new_mask[int(sample_index)] = bool(decision.created_new_prototype)

    refreshed = [
        refresh_prototype(
            prototype=prototype,
            embeddings=embeddings,
            config=config,
            base_threshold=float(calibration.tau_base),
            threshold_floor=float(calibration.tau_floor),
        )
        for prototype in prototypes
    ]

    merge_result = merge_prototypes(
        prototypes=refreshed,
        embeddings=embeddings,
        base_threshold=float(calibration.tau_base),
        config=config,
    )

    final_thresholds = np.zeros((num_samples,), dtype=np.float32)
    final_confidences = np.zeros((num_samples,), dtype=np.float32)
    final_prototype_ids = merge_result.prototype_ids.astype(np.int64)
    prototype_lookup = {int(prototype.id): prototype for prototype in merge_result.prototypes}
    for sample_index, prototype_id in enumerate(final_prototype_ids.tolist()):
        prototype = prototype_lookup[int(prototype_id)]
        final_thresholds[int(sample_index)] = float(prototype.assign_threshold)
        final_confidences[int(sample_index)] = float(prototype.confidence)

    prototype_export = export_prototypes(
        prototypes=merge_result.prototypes,
        max_supports=int(config.prototype.max_supports),
        embedding_dim=int(embeddings.shape[1]),
    )

    return {
        'ordered_indices': ordered_indices,
        'labels': merge_result.labels.astype(np.int64),
        'prototype_ids': final_prototype_ids,
        'labels_before_merge': labels_before_merge,
        'prototype_ids_before_merge': prototype_ids_before_merge,
        'assignment_scores': assignment_scores,
        'assignment_margins': assignment_margins,
        'assignment_thresholds': final_thresholds,
        'assignment_thresholds_before_merge': assignment_thresholds,
        'center_scores': center_scores,
        'support_scores': support_scores,
        'prototype_confidences': final_confidences,
        'created_new_mask': created_new_mask.astype(np.bool_),
        'num_prototypes_before_merge': np.int64(len(refreshed)),
        'num_prototypes_after_merge': np.int64(merge_result.num_prototypes),
        'num_merge_events': np.int64(merge_result.num_merge_events),
        'prototype_export_ids': prototype_export.ids,
        'prototype_export_centers': prototype_export.centers,
        'prototype_export_thresholds': prototype_export.thresholds,
        'prototype_export_confidences': prototype_export.confidences,
        'prototype_export_sizes': prototype_export.sizes,
        'prototype_export_support_indices': prototype_export.support_indices,
        'prototype_export_member_counts': prototype_export.member_counts,
    }
