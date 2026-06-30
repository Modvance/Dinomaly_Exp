from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SemanticPrototype:
    id: int
    center: np.ndarray
    members: List[int] = field(default_factory=list)
    supports: List[int] = field(default_factory=list)
    assign_threshold: float = 0.0
    confidence: float = 0.0
    member_scores: List[float] = field(default_factory=list)
    member_reliabilities: List[float] = field(default_factory=list)


@dataclass
class PrototypeScore:
    prototype_id: int
    center_score: float
    support_score: float
    score: float
    threshold: float
    confidence: float


@dataclass
class AssignmentDecision:
    assigned: bool
    prototype_id: Optional[int]
    assignment_score: float
    assignment_margin: float
    assignment_threshold: float
    center_score: Optional[float]
    support_score: Optional[float]
    created_new_prototype: bool


@dataclass
class PrototypeExport:
    ids: np.ndarray
    centers: np.ndarray
    thresholds: np.ndarray
    confidences: np.ndarray
    sizes: np.ndarray
    support_indices: np.ndarray
    member_counts: np.ndarray


@dataclass
class MergeResult:
    labels: np.ndarray
    prototype_ids: np.ndarray
    num_merge_events: int
    num_prototypes: int
    prototypes: List[SemanticPrototype]


def l2_normalize(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = float(np.linalg.norm(array))
    if norm <= float(eps):
        return array.astype(np.float32)
    return (array / norm).astype(np.float32)


def _score_to_supports(embedding: np.ndarray, prototype: SemanticPrototype, embeddings: np.ndarray, top_r: int) -> float:
    if len(prototype.supports) == 0:
        return float(np.dot(embedding, prototype.center))
    scores = np.asarray([
        float(np.dot(embedding, embeddings[int(member_index)]))
        for member_index in prototype.supports
    ], dtype=np.float32)
    top_k = min(int(top_r), int(scores.size))
    if top_k <= 0:
        return float(np.dot(embedding, prototype.center))
    order = np.sort(scores)[-top_k:]
    return float(np.mean(order))


def score_prototype(embedding: np.ndarray, prototype: SemanticPrototype, embeddings: np.ndarray, top_r: int) -> PrototypeScore:
    center_score = float(np.dot(embedding, prototype.center))
    support_score = _score_to_supports(embedding, prototype, embeddings, top_r=top_r)
    score = float(max(center_score, support_score))
    return PrototypeScore(
        prototype_id=int(prototype.id),
        center_score=float(center_score),
        support_score=float(support_score),
        score=float(score),
        threshold=float(prototype.assign_threshold),
        confidence=float(prototype.confidence),
    )


def select_supports(member_indices: List[int], member_scores: List[float], max_supports: int) -> List[int]:
    if len(member_indices) == 0:
        return []
    ranked = sorted(
        zip(member_indices, member_scores),
        key=lambda item: (-float(item[1]), int(item[0])),
    )
    return [int(member_index) for member_index, _ in ranked[:max(1, int(max_supports))]]


def compute_assign_threshold(prototype: SemanticPrototype, config, base_threshold: float, threshold_floor: float) -> float:
    if not bool(config.assignment.use_class_adaptive_threshold):
        return float(base_threshold)
    min_members = int(config.prototype.min_members_for_adaptive_threshold)
    if len(prototype.member_scores) < min_members:
        return float(base_threshold)
    adaptive = float(np.quantile(
        np.asarray(prototype.member_scores, dtype=np.float32),
        float(config.prototype.member_score_quantile),
    ))
    return float(max(float(threshold_floor), min(float(base_threshold), adaptive)))


def compute_prototype_confidence(prototype: SemanticPrototype) -> float:
    if len(prototype.member_scores) == 0:
        return 0.0
    member_score = float(np.mean(np.asarray(prototype.member_scores, dtype=np.float32)))
    reliability_score = float(np.mean(np.asarray(prototype.member_reliabilities, dtype=np.float32))) if len(prototype.member_reliabilities) > 0 else member_score
    return float(0.5 * member_score + 0.5 * reliability_score)


def create_prototype(prototype_id: int, sample_index: int, embeddings: np.ndarray, reliability: float,
                     base_threshold: float, config) -> SemanticPrototype:
    center = l2_normalize(embeddings[int(sample_index)])
    prototype = SemanticPrototype(
        id=int(prototype_id),
        center=center,
        members=[int(sample_index)],
        supports=[int(sample_index)],
        assign_threshold=float(base_threshold),
        confidence=float(reliability),
        member_scores=[1.0],
        member_reliabilities=[float(reliability)],
    )
    prototype.supports = select_supports(prototype.members, prototype.member_scores, int(config.prototype.max_supports))
    prototype.confidence = compute_prototype_confidence(prototype)
    return prototype


def update_prototype(prototype: SemanticPrototype, sample_index: int, embedding: np.ndarray, reliability: float,
                     assignment_score: float, assignment_margin: float, config, base_threshold: float,
                     threshold_floor: float, embeddings: np.ndarray) -> SemanticPrototype:
    prototype.members.append(int(sample_index))
    prototype.member_scores.append(float(assignment_score))
    prototype.member_reliabilities.append(float(reliability))

    should_update_center = True
    if bool(config.prototype.update_center_only_high_confidence):
        bonus = float(config.prototype.high_confidence_margin_bonus)
        should_update_center = float(assignment_margin) >= float(config.assignment.margin_threshold) + bonus
    if should_update_center:
        member_embeddings = embeddings_from_indices(prototype.members, embedding_source=embeddings)
        center = np.mean(member_embeddings, axis=0)
        prototype.center = l2_normalize(center)

    prototype.supports = select_supports(prototype.members, prototype.member_scores, int(config.prototype.max_supports))
    prototype.assign_threshold = compute_assign_threshold(prototype, config, base_threshold=base_threshold, threshold_floor=threshold_floor)
    prototype.confidence = compute_prototype_confidence(prototype)
    return prototype


def embeddings_from_indices(member_indices: List[int], embedding_source: Optional[np.ndarray], fallback_embedding: Optional[np.ndarray] = None) -> np.ndarray:
    if embedding_source is None:
        if fallback_embedding is None:
            raise ValueError('embedding_source or fallback_embedding is required')
        if len(member_indices) == 1:
            return np.asarray([fallback_embedding], dtype=np.float32)
        raise ValueError('embedding_source is required for multi-member prototype updates')
    return np.asarray([embedding_source[int(member_index)] for member_index in member_indices], dtype=np.float32)


def refresh_prototype(prototype: SemanticPrototype, embeddings: np.ndarray, config, base_threshold: float,
                      threshold_floor: float) -> SemanticPrototype:
    member_embeddings = embeddings_from_indices(prototype.members, embedding_source=embeddings)
    prototype.center = l2_normalize(np.mean(member_embeddings, axis=0))
    prototype.supports = select_supports(prototype.members, prototype.member_scores, int(config.prototype.max_supports))
    prototype.assign_threshold = compute_assign_threshold(prototype, config, base_threshold=base_threshold, threshold_floor=threshold_floor)
    prototype.confidence = compute_prototype_confidence(prototype)
    return prototype


def assign_to_best_prototype(sample_index: int, embedding: np.ndarray, reliability: float, prototypes: List[SemanticPrototype],
                             embeddings: np.ndarray, config, base_threshold: float, threshold_floor: float) -> AssignmentDecision:
    if len(prototypes) == 0:
        return AssignmentDecision(
            assigned=False,
            prototype_id=None,
            assignment_score=1.0,
            assignment_margin=1.0,
            assignment_threshold=float(base_threshold),
            center_score=None,
            support_score=None,
            created_new_prototype=True,
        )

    scores = [score_prototype(embedding, prototype, embeddings, top_r=int(config.prototype.support_top_r)) for prototype in prototypes]
    scores.sort(key=lambda item: (-float(item.score), int(item.prototype_id)))
    best = scores[0]
    second = scores[1] if len(scores) > 1 else None
    margin = float(best.score - second.score) if second is not None else 1.0
    passes_threshold = float(best.score) >= float(best.threshold)
    passes_margin = float(margin) >= float(config.assignment.margin_threshold)
    if passes_threshold and passes_margin:
        return AssignmentDecision(
            assigned=True,
            prototype_id=int(best.prototype_id),
            assignment_score=float(best.score),
            assignment_margin=float(margin),
            assignment_threshold=float(best.threshold),
            center_score=float(best.center_score),
            support_score=float(best.support_score),
            created_new_prototype=False,
        )
    return AssignmentDecision(
        assigned=False,
        prototype_id=None,
        assignment_score=float(best.score),
        assignment_margin=float(margin),
        assignment_threshold=float(best.threshold),
        center_score=float(best.center_score),
        support_score=float(best.support_score),
        created_new_prototype=True,
    )


def merge_prototypes(prototypes: List[SemanticPrototype], embeddings: np.ndarray, base_threshold: float, config) -> MergeResult:
    if len(prototypes) <= 1 or not bool(config.merge.enabled):
        labels = np.full((embeddings.shape[0],), -1, dtype=np.int64)
        prototype_ids = np.full((embeddings.shape[0],), -1, dtype=np.int64)
        for contiguous_index, prototype in enumerate(sorted(prototypes, key=lambda item: int(item.id))):
            for member_index in prototype.members:
                labels[int(member_index)] = int(contiguous_index)
                prototype_ids[int(member_index)] = int(prototype.id)
        return MergeResult(
            labels=labels,
            prototype_ids=prototype_ids,
            num_merge_events=0,
            num_prototypes=len(prototypes),
            prototypes=sorted(prototypes, key=lambda item: int(item.id)),
        )

    working = [prototype for prototype in prototypes]
    merge_threshold = float(base_threshold + float(config.merge.merge_threshold_offset))
    merge_events = 0

    for _ in range(max(1, int(config.merge.max_merge_passes))):
        merged_any = False
        consumed = set()
        next_prototypes: List[SemanticPrototype] = []
        ordered = sorted(working, key=lambda item: int(item.id))
        for left_index, left in enumerate(ordered):
            if left_index in consumed:
                continue
            current = left
            for right_index in range(left_index + 1, len(ordered)):
                if right_index in consumed:
                    continue
                right = ordered[right_index]
                center_score = float(np.dot(current.center, right.center))
                support_scores = []
                for left_support in current.supports:
                    for right_support in right.supports:
                        support_scores.append(float(np.dot(embeddings[int(left_support)], embeddings[int(right_support)])))
                if len(support_scores) > 0:
                    top_r = max(1, int(config.merge.support_top_r))
                    support_scores = sorted(support_scores)[-top_r:]
                    support_score = float(np.mean(support_scores))
                else:
                    support_score = center_score
                score = max(center_score, support_score)
                if float(score) >= merge_threshold:
                    consumed.add(right_index)
                    current.members = sorted(set(current.members + right.members))
                    current.member_scores.extend(right.member_scores)
                    current.member_reliabilities.extend(right.member_reliabilities)
                    current = refresh_prototype(current, embeddings, config, base_threshold=base_threshold, threshold_floor=base_threshold)
                    merge_events += 1
                    merged_any = True
            next_prototypes.append(current)
        working = next_prototypes
        if not merged_any:
            break

    labels = np.full((embeddings.shape[0],), -1, dtype=np.int64)
    prototype_ids = np.full((embeddings.shape[0],), -1, dtype=np.int64)
    ordered = sorted(working, key=lambda item: int(item.id))
    for contiguous_index, prototype in enumerate(ordered):
        refresh_prototype(prototype, embeddings, config, base_threshold=base_threshold, threshold_floor=base_threshold)
        for member_index in prototype.members:
            labels[int(member_index)] = int(contiguous_index)
            prototype_ids[int(member_index)] = int(prototype.id)
    return MergeResult(
        labels=labels,
        prototype_ids=prototype_ids,
        num_merge_events=int(merge_events),
        num_prototypes=int(len(ordered)),
        prototypes=ordered,
    )


def export_prototypes(prototypes: List[SemanticPrototype], max_supports: int, embedding_dim: int) -> PrototypeExport:
    ordered = sorted(prototypes, key=lambda item: int(item.id))
    ids = np.asarray([int(prototype.id) for prototype in ordered], dtype=np.int64)
    centers = np.asarray([prototype.center for prototype in ordered], dtype=np.float32) if len(ordered) > 0 else np.zeros((0, int(embedding_dim)), dtype=np.float32)
    thresholds = np.asarray([float(prototype.assign_threshold) for prototype in ordered], dtype=np.float32)
    confidences = np.asarray([float(prototype.confidence) for prototype in ordered], dtype=np.float32)
    sizes = np.asarray([int(len(prototype.members)) for prototype in ordered], dtype=np.int64)
    member_counts = sizes.copy()
    support_indices = np.full((len(ordered), max(1, int(max_supports))), -1, dtype=np.int64)
    for row_index, prototype in enumerate(ordered):
        for col_index, sample_index in enumerate(prototype.supports[:max(1, int(max_supports))]):
            support_indices[row_index, col_index] = int(sample_index)
    return PrototypeExport(
        ids=ids,
        centers=centers,
        thresholds=thresholds,
        confidences=confidences,
        sizes=sizes,
        support_indices=support_indices,
        member_counts=member_counts,
    )


def build_prototype_lookup(prototypes: List[SemanticPrototype]) -> Dict[int, SemanticPrototype]:
    return {int(prototype.id): prototype for prototype in prototypes}
