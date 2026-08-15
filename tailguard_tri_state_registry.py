"""Train-only tri-state reconciliation registry for TailGuard replay.

The legacy pseudo-class registry assigns one class to each retained sample.
That representation couples structural attachment, routing, and patch-memory
eligibility.  This module keeps those relations in separate tables so a
head-affiliated candidate can contribute to a head prototype and a routing
shield without becoming a normal patch-memory reference.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


TRAIN_ONLY_LOO_COLUMNS = (
    'sample_idx',
    'pseudo_class_id',
    'source_status',
    'loo_predicted_pseudo_class_id',
    'loo_margin',
    'loo_status',
)

SAMPLE_ROLE_COLUMNS = (
    'sample_idx',
    'class_id',
    'base_idx',
    'sample_key',
    'img_path',
    'source_status',
    'initial_support_role',
    'rgd_role',
    'final_structural_role',
    'routing_role',
    'memory_role',
    'memory_eligible',
    'cross_view_status',
    'loo_margin',
    'low_margin',
    'decision_reason',
    'source_full_pseudo_class_id',
    'source_raw_pseudo_class_id',
)

ROUTING_CLASS_COLUMNS = (
    'route_class_id',
    'route_class_key',
    'route_class_type',
    'pseudo_class_type',
    'fallback_behavior',
    'memory_eligible',
    'source_partition',
    'source_partition_id',
    'num_routing_members',
    'num_memory_members',
)

ROUTING_MEMBER_COLUMNS = (
    'sample_idx',
    'class_id',
    'base_idx',
    'route_class_id',
    'route_class_type',
    'contribution_role',
)

MEMORY_MEMBER_COLUMNS = (
    'sample_idx',
    'class_id',
    'base_idx',
    'route_class_id',
    'route_class_type',
    'memory_role',
)

PROTOCOL_KINDS = {
    'independent_development',
    'pre_registered',
    'exploratory_target_audited',
}


@dataclass(frozen=True)
class TriStatePolicy:
    """Configuration whose decisions use only saved training-side evidence."""

    enable_auxiliary_memory: bool = True
    low_margin_threshold: Optional[float] = None
    threshold_protocol_id: Optional[str] = None
    threshold_protocol_kind: Optional[str] = None

    def validate(self):
        if self.low_margin_threshold is None:
            if self.threshold_protocol_id is not None or self.threshold_protocol_kind is not None:
                raise ValueError('threshold protocol metadata requires a low-margin threshold')
            return
        threshold = float(self.low_margin_threshold)
        if not np.isfinite(threshold) or threshold < 0.0 or threshold > 2.0:
            raise ValueError('low-margin threshold must be finite and in [0, 2]')
        if not str(self.threshold_protocol_id or '').strip():
            raise ValueError('a low-margin threshold requires threshold_protocol_id')
        if self.threshold_protocol_kind not in PROTOCOL_KINDS:
            raise ValueError(
                'threshold_protocol_kind must be one of: {}'.format(
                    ', '.join(sorted(PROTOCOL_KINDS))
                )
            )

    def as_dict(self):
        self.validate()
        threshold = None if self.low_margin_threshold is None else float(self.low_margin_threshold)
        return {
            'enable_auxiliary_memory': bool(self.enable_auxiliary_memory),
            'low_margin_threshold': threshold,
            'threshold_protocol_id': self.threshold_protocol_id,
            'threshold_protocol_kind': self.threshold_protocol_kind,
            'runtime_uses_labels': False,
            'runtime_evidence': list(TRAIN_ONLY_LOO_COLUMNS),
            'paper_eligible_threshold_protocol': bool(
                threshold is None
                or self.threshold_protocol_kind in {'independent_development', 'pre_registered'}
            ),
        }


def _require_columns(frame: pd.DataFrame, required, name: str):
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError('{} is missing required columns: {}'.format(name, ', '.join(missing)))


def _validated_members(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {
        'sample_idx', 'class_id', 'base_idx', 'pseudo_class_id',
        'pseudo_class_type', 'source_status',
    }
    _require_columns(frame, required, name)
    output = frame.copy().reset_index(drop=True)
    for column in ('sample_idx', 'class_id', 'base_idx', 'pseudo_class_id'):
        output[column] = output[column].astype(int)
    if output['sample_idx'].duplicated().any():
        raise ValueError('{} contains duplicate sample_idx values'.format(name))
    if output.duplicated(['class_id', 'base_idx']).any():
        raise ValueError('{} contains duplicate class_id/base_idx identities'.format(name))
    if 'sample_key' not in output.columns:
        output['sample_key'] = output['sample_idx']
    if 'img_path' not in output.columns:
        output['img_path'] = ''
    output['sample_key'] = output['sample_key'].astype(int)
    output['img_path'] = output['img_path'].astype(str)
    return output


def _validate_classes(members: pd.DataFrame,
                      classes: Optional[pd.DataFrame],
                      name: str):
    if classes is None:
        return
    _require_columns(classes, {'pseudo_class_id', 'pseudo_class_type', 'num_members'}, name)
    class_frame = classes.copy()
    class_frame['pseudo_class_id'] = class_frame['pseudo_class_id'].astype(int)
    if class_frame['pseudo_class_id'].duplicated().any():
        raise ValueError('{} contains duplicate pseudo_class_id values'.format(name))
    actual = members.groupby('pseudo_class_id').size().to_dict()
    expected = class_frame.set_index('pseudo_class_id')['num_members'].astype(int).to_dict()
    if actual != expected:
        raise ValueError('{} membership counts do not match its class table'.format(name))
    member_types = members.groupby('pseudo_class_id')['pseudo_class_type'].first().to_dict()
    class_types = class_frame.set_index('pseudo_class_id')['pseudo_class_type'].astype(str).to_dict()
    if member_types != class_types:
        raise ValueError('{} member and class types do not agree'.format(name))


def _validate_shared_retained_set(full_members: pd.DataFrame,
                                  raw_members: pd.DataFrame):
    full_identity = full_members.set_index('sample_idx')[['class_id', 'base_idx']].sort_index()
    raw_identity = raw_members.set_index('sample_idx')[['class_id', 'base_idx']].sort_index()
    if not full_identity.equals(raw_identity):
        raise ValueError('Full and Raw-E registries do not share the same retained identities')


def _train_only_attached_evidence(full_members: pd.DataFrame,
                                  loo_report: pd.DataFrame) -> pd.DataFrame:
    # Slice the allowlist before making decisions. Oracle columns may coexist in
    # the saved diagnostic CSV, but cannot flow into this registry builder.
    _require_columns(loo_report, TRAIN_ONLY_LOO_COLUMNS, 'CLS leave-one-out report')
    report = loo_report.loc[:, TRAIN_ONLY_LOO_COLUMNS].copy().reset_index(drop=True)
    report['sample_idx'] = report['sample_idx'].astype(int)
    if report['sample_idx'].duplicated().any():
        raise ValueError('CLS leave-one-out report contains duplicate sample_idx values')
    attached = full_members.loc[
        full_members['source_status'].astype(str) == 'T_head_normal',
        ['sample_idx', 'pseudo_class_id'],
    ].rename(columns={'pseudo_class_id': 'expected_pseudo_class_id'})
    evidence = attached.merge(report, on='sample_idx', how='left', validate='one_to_one')
    if evidence[list(TRAIN_ONLY_LOO_COLUMNS[1:])].isna().all(axis=1).any():
        missing = evidence.loc[
            evidence[list(TRAIN_ONLY_LOO_COLUMNS[1:])].isna().all(axis=1), 'sample_idx'
        ].astype(int).tolist()
        raise ValueError('CLS leave-one-out report is missing attached samples: {}'.format(missing[:10]))
    if not (
        evidence['source_status'].astype(str) == 'T_head_normal'
    ).all():
        raise ValueError('CLS leave-one-out report has inconsistent attached source_status')
    reported_class = pd.to_numeric(evidence['pseudo_class_id'], errors='coerce')
    if reported_class.isna().any() or not (
        reported_class.astype(int) == evidence['expected_pseudo_class_id'].astype(int)
    ).all():
        raise ValueError('CLS leave-one-out report does not match the Full registry')
    return evidence


def classify_attached_candidates(full_members: pd.DataFrame,
                                 loo_report: pd.DataFrame,
                                 policy: TriStatePolicy) -> pd.DataFrame:
    """Classify RGD-attached samples without reading labels or test results."""
    policy.validate()
    full = _validated_members(full_members, 'Full members')
    evidence = _train_only_attached_evidence(full, loo_report)
    predicted = pd.to_numeric(evidence['loo_predicted_pseudo_class_id'], errors='coerce')
    margins = pd.to_numeric(evidence['loo_margin'], errors='coerce')
    valid = (
        evidence['loo_status'].astype(str).eq('evaluated')
        & np.isfinite(predicted.to_numpy(dtype=float))
        & np.isfinite(margins.to_numpy(dtype=float))
    )
    agrees = valid & (
        predicted.fillna(-1).astype(int) == evidence['expected_pseudo_class_id'].astype(int)
    )
    if policy.low_margin_threshold is None:
        low_margin = pd.Series(False, index=evidence.index)
    else:
        low_margin = valid & (margins < float(policy.low_margin_threshold))
    auxiliary = valid & (~agrees | low_margin) & bool(policy.enable_auxiliary_memory)

    reasons = np.full(len(evidence), 'cross_view_agreement', dtype=object)
    reasons[~valid.to_numpy()] = 'invalid_loo_fallback_shield'
    reasons[(valid & ~agrees & ~low_margin).to_numpy()] = 'cross_view_disagreement'
    reasons[(valid & agrees & low_margin).to_numpy()] = 'low_cls_margin'
    reasons[(valid & ~agrees & low_margin).to_numpy()] = 'disagreement_and_low_margin'
    if not policy.enable_auxiliary_memory:
        reasons[(valid & (~agrees | low_margin)).to_numpy()] = 'auxiliary_disabled_fallback_shield'

    output = evidence[['sample_idx', 'expected_pseudo_class_id']].copy()
    output['cross_view_status'] = np.where(
        ~valid, 'invalid', np.where(agrees, 'agrees', 'disagrees')
    )
    output['loo_margin'] = margins.astype(float)
    output['low_margin'] = low_margin.astype(bool)
    output['auxiliary_memory_eligible'] = auxiliary.astype(bool)
    output['decision_reason'] = reasons
    return output.sort_values('sample_idx', kind='mergesort').reset_index(drop=True)


def build_tri_state_registry(full_members: pd.DataFrame,
                             raw_members: pd.DataFrame,
                             loo_report: pd.DataFrame,
                             policy: TriStatePolicy,
                             full_classes: Optional[pd.DataFrame] = None,
                             raw_classes: Optional[pd.DataFrame] = None) -> Dict:
    """Build separate sample-role, routing, and memory membership tables."""
    policy.validate()
    full = _validated_members(full_members, 'Full members')
    raw = _validated_members(raw_members, 'Raw-E members')
    _validate_classes(full, full_classes, 'Full classes')
    _validate_classes(raw, raw_classes, 'Raw-E classes')
    _validate_shared_retained_set(full, raw)

    allowed_full_statuses = {'H_clean', 'T_head_normal', 'T_open'}
    unexpected_statuses = sorted(set(full['source_status'].astype(str)).difference(allowed_full_statuses))
    if unexpected_statuses:
        raise ValueError('Full registry has unsupported source_status values: {}'.format(unexpected_statuses))
    raw_by_sample = raw.set_index('sample_idx', drop=False)
    tail_ids = set(full.loc[full['source_status'] != 'H_clean', 'sample_idx'].astype(int))
    if not (raw_by_sample.loc[sorted(tail_ids), 'pseudo_class_type'].astype(str) == 'tail').all():
        raise ValueError('every initial tail candidate must be a Raw-E tail member')

    decisions = classify_attached_candidates(full, loo_report, policy).set_index('sample_idx')
    role_rows = []
    for row in full.sort_values('sample_idx', kind='mergesort').itertuples(index=False):
        sample_idx = int(row.sample_idx)
        source_status = str(row.source_status)
        raw_class_id = int(raw_by_sample.loc[sample_idx, 'pseudo_class_id'])
        common = {
            'sample_idx': sample_idx,
            'class_id': int(row.class_id),
            'base_idx': int(row.base_idx),
            'sample_key': int(row.sample_key),
            'img_path': str(row.img_path),
            'source_status': source_status,
            'source_full_pseudo_class_id': int(row.pseudo_class_id),
            'source_raw_pseudo_class_id': raw_class_id,
        }
        if source_status == 'H_clean':
            role_rows.append({
                **common,
                'initial_support_role': 'head_candidate',
                'rgd_role': 'not_applicable',
                'final_structural_role': 'supported_head',
                'routing_role': 'head',
                'memory_role': 'none',
                'memory_eligible': False,
                'cross_view_status': 'not_applicable',
                'loo_margin': np.nan,
                'low_margin': False,
                'decision_reason': 'purified_head_member',
            })
        elif source_status == 'T_open':
            role_rows.append({
                **common,
                'initial_support_role': 'tail_candidate',
                'rgd_role': 'open_tail',
                'final_structural_role': 'open_tail',
                'routing_role': 'open_tail',
                'memory_role': 'open_tail_memory',
                'memory_eligible': True,
                'cross_view_status': 'not_applicable',
                'loo_margin': np.nan,
                'low_margin': False,
                'decision_reason': 'rgd_open_tail',
            })
        else:
            decision = decisions.loc[sample_idx]
            auxiliary = bool(decision['auxiliary_memory_eligible'])
            role_rows.append({
                **common,
                'initial_support_role': 'tail_candidate',
                'rgd_role': 'head_affiliated',
                'final_structural_role': 'residual_tail' if auxiliary else 'head_affiliated',
                'routing_role': 'auxiliary_tail' if auxiliary else 'head_and_shield',
                'memory_role': 'auxiliary_tail_memory' if auxiliary else 'none',
                'memory_eligible': auxiliary,
                'cross_view_status': str(decision['cross_view_status']),
                'loo_margin': float(decision['loo_margin']),
                'low_margin': bool(decision['low_margin']),
                'decision_reason': str(decision['decision_reason']),
            })
    roles = pd.DataFrame(role_rows, columns=SAMPLE_ROLE_COLUMNS)

    route_class_rows = []
    route_member_rows = []
    memory_member_rows = []

    def add_route_class(class_key,
                        class_type,
                        compatible_type,
                        member_ids,
                        memory_ids,
                        source_partition,
                        source_partition_id,
                        contribution_role,
                        memory_role):
        member_ids = sorted(set(int(value) for value in member_ids))
        memory_ids = sorted(set(int(value) for value in memory_ids))
        if not member_ids:
            return
        if not set(memory_ids).issubset(member_ids):
            raise ValueError('memory members must also contribute to their routing prototype')
        route_class_id = len(route_class_rows)
        memory_enabled = bool(memory_ids)
        route_class_rows.append({
            'route_class_id': route_class_id,
            'route_class_key': str(class_key),
            'route_class_type': str(class_type),
            'pseudo_class_type': str(compatible_type),
            'fallback_behavior': 'fuse_memory' if memory_enabled else 'reconstruction',
            'memory_eligible': memory_enabled,
            'source_partition': str(source_partition),
            'source_partition_id': int(source_partition_id),
            'num_routing_members': len(member_ids),
            'num_memory_members': len(memory_ids),
        })
        identity = roles.set_index('sample_idx')
        for sample_idx in member_ids:
            route_member_rows.append({
                'sample_idx': sample_idx,
                'class_id': int(identity.loc[sample_idx, 'class_id']),
                'base_idx': int(identity.loc[sample_idx, 'base_idx']),
                'route_class_id': route_class_id,
                'route_class_type': str(class_type),
                'contribution_role': str(contribution_role),
            })
        for sample_idx in memory_ids:
            memory_member_rows.append({
                'sample_idx': sample_idx,
                'class_id': int(identity.loc[sample_idx, 'class_id']),
                'base_idx': int(identity.loc[sample_idx, 'base_idx']),
                'route_class_id': route_class_id,
                'route_class_type': str(class_type),
                'memory_role': str(memory_role),
            })

    # Supported head prototypes retain only reconciled affiliated candidates.
    head_roles = roles.loc[
        roles['final_structural_role'].isin(['supported_head', 'head_affiliated'])
    ]
    for source_id, group in head_roles.groupby('source_full_pseudo_class_id', sort=True):
        add_route_class(
            'head:{}'.format(int(source_id)), 'head', 'head',
            group['sample_idx'], [], 'full_head', source_id,
            'head_prototype', 'none',
        )

    reliable_attached = roles.loc[roles['routing_role'] == 'head_and_shield']
    # A shield is deliberately singleton: averaging unrelated attached
    # candidates can leave a gap around exactly the rare mode it must protect.
    for row in reliable_attached.sort_values('sample_idx', kind='mergesort').itertuples(index=False):
        sample_idx = int(row.sample_idx)
        add_route_class(
            'shield:{}'.format(sample_idx), 'shield', 'head',
            [sample_idx], [], 'attached_sample', sample_idx,
            'shield_prototype', 'none',
        )

    open_tail = roles.loc[roles['routing_role'] == 'open_tail']
    for source_id, group in open_tail.groupby('source_full_pseudo_class_id', sort=True):
        add_route_class(
            'open_tail:{}'.format(int(source_id)), 'open_tail', 'tail',
            group['sample_idx'], group['sample_idx'], 'full_open_tail', source_id,
            'open_tail_prototype', 'open_tail_memory',
        )

    auxiliary = roles.loc[roles['routing_role'] == 'auxiliary_tail']
    for source_id, group in auxiliary.groupby('source_raw_pseudo_class_id', sort=True):
        add_route_class(
            'auxiliary_tail:{}'.format(int(source_id)), 'auxiliary_tail', 'tail',
            group['sample_idx'], group['sample_idx'], 'raw_tail_component', source_id,
            'auxiliary_tail_prototype', 'auxiliary_tail_memory',
        )

    route_classes = pd.DataFrame(route_class_rows, columns=ROUTING_CLASS_COLUMNS)
    routing_members = pd.DataFrame(route_member_rows, columns=ROUTING_MEMBER_COLUMNS)
    memory_members = pd.DataFrame(memory_member_rows, columns=MEMORY_MEMBER_COLUMNS)
    _validate_tri_state_integrity(roles, route_classes, routing_members, memory_members)

    policy_payload = policy.as_dict()
    summary = {
        'status': 'completed',
        'runtime_decision_is_train_only': True,
        'runtime_evidence_columns': list(TRAIN_ONLY_LOO_COLUMNS),
        'oracle_columns_consumed': [],
        'num_retained_samples': int(len(roles)),
        'num_routing_classes': int(len(route_classes)),
        'num_head_classes': int((route_classes['route_class_type'] == 'head').sum()),
        'num_shield_classes': int((route_classes['route_class_type'] == 'shield').sum()),
        'num_open_tail_classes': int((route_classes['route_class_type'] == 'open_tail').sum()),
        'num_auxiliary_tail_classes': int((route_classes['route_class_type'] == 'auxiliary_tail').sum()),
        'num_head_affiliated': int((roles['rgd_role'] == 'head_affiliated').sum()),
        'num_cross_view_disagreements': int((roles['cross_view_status'] == 'disagrees').sum()),
        'num_low_margin': int(roles['low_margin'].sum()),
        'num_auxiliary_memory_samples': int((roles['memory_role'] == 'auxiliary_tail_memory').sum()),
        'num_open_tail_memory_samples': int((roles['memory_role'] == 'open_tail_memory').sum()),
        'policy': policy_payload,
        'integrity': {
            'sample_roles_cover_retained_once': True,
            'every_sample_has_a_route': True,
            'memory_members_are_unique': True,
            'shield_has_no_memory': True,
            'structural_and_memory_roles_are_separate': True,
        },
    }
    return {
        'sample_roles_df': roles,
        'routing_classes_df': route_classes,
        'routing_members_df': routing_members,
        'memory_members_df': memory_members,
        'summary': summary,
    }


def _validate_tri_state_integrity(roles: pd.DataFrame,
                                  route_classes: pd.DataFrame,
                                  routing_members: pd.DataFrame,
                                  memory_members: pd.DataFrame):
    if roles['sample_idx'].duplicated().any():
        raise ValueError('tri-state sample roles must be unique')
    retained_ids = set(roles['sample_idx'].astype(int))
    routed_ids = set(routing_members['sample_idx'].astype(int))
    if retained_ids != routed_ids:
        raise ValueError('tri-state routing membership must cover every retained sample')
    if memory_members['sample_idx'].duplicated().any():
        raise ValueError('a sample cannot belong to more than one patch-memory bank')
    memory_ids = set(memory_members['sample_idx'].astype(int))
    eligible_ids = set(roles.loc[roles['memory_eligible'], 'sample_idx'].astype(int))
    if memory_ids != eligible_ids:
        raise ValueError('sample memory eligibility does not match memory membership')
    if len(route_classes) == 0 or route_classes['route_class_id'].duplicated().any():
        raise ValueError('tri-state routing classes are empty or duplicated')
    class_ids = set(route_classes['route_class_id'].astype(int))
    if set(routing_members['route_class_id'].astype(int)) != class_ids:
        raise ValueError('every routing class must have routing members')
    if not set(memory_members['route_class_id'].astype(int)).issubset(class_ids):
        raise ValueError('memory membership references an unknown routing class')
    if (memory_members['route_class_type'] == 'shield').any():
        raise ValueError('routing shields must never own patch memory')
