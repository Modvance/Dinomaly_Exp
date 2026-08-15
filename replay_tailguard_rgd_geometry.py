"""Replay TailGuard RGD with unpurified and purified supported geometry.

This is an analysis-only counterfactual: it reuses saved embeddings, support
assignments, and H decisions. It never loads or trains a detector.
"""

import argparse
import csv
import hashlib
import json
import os
from types import SimpleNamespace

import pandas as pd
import torch

from tailguard_attachment import build_tail_attachment_plan


SCENARIOS = (
    'mvtec_pareto_seed01',
    'mvtec_step_k4_seed01',
    'mvtec_step_k1_seed01',
    'visa_pareto_seed01',
    'visa_step_k4_seed01',
    'visa_step_k1_seed01',
)
BRANCHES = ('unpurified_h0', 'purified_hc')


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)


def _read_csv(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _load_embeddings(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location='cpu')
    required = {'sample_idx', 'embeddings', 'embedding_source'}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError('invalid grouping embeddings payload: {}'.format(path))
    return payload


def _analysis_labels(details):
    required = {'sample_idx', 'tail_candidate', 'is_contaminated', 'is_gt_tail'}
    missing = required.difference(details.columns)
    if missing:
        raise ValueError('analysis detail columns are missing: {}'.format(sorted(missing)))
    labels = details[list(required)].copy()
    for column in required:
        labels[column] = labels[column].astype(int)
    if labels['sample_idx'].duplicated().any():
        raise ValueError('analysis details contain duplicate sample_idx')
    return labels


def _head_frames(head_decisions):
    required = {'sample_idx', 'group_id', 'tail_candidate', 'prune_selected'}
    missing = required.difference(head_decisions.columns)
    if missing:
        raise ValueError('H decision columns are missing: {}'.format(sorted(missing)))
    head = head_decisions.loc[head_decisions['tail_candidate'].astype(int) == 0].copy()
    if head['sample_idx'].duplicated().any():
        raise ValueError('H decisions contain duplicate sample_idx')
    raw = head.copy().reset_index(drop=True)
    clean = head.loc[~head['prune_selected'].astype(bool)].copy().reset_index(drop=True)
    return raw, clean


def _composition(frame):
    return {
        'num_samples': int(len(frame)),
        'clean_gt_tail': int(((frame['is_contaminated'] == 0) & (frame['is_gt_tail'] == 1)).sum()),
        'clean_gt_head': int(((frame['is_contaminated'] == 0) & (frame['is_gt_tail'] == 0)).sum()),
        'contamination': int((frame['is_contaminated'] == 1).sum()),
    }


def _branch_summary(scenario, geometry_name, plan, labels):
    analysis_columns = ['sample_idx', 'is_contaminated', 'is_gt_tail']
    open_frame = plan['tail_open_df'].drop(
        columns=['is_contaminated', 'is_gt_tail'], errors='ignore'
    ).merge(labels[analysis_columns], on='sample_idx', how='left', validate='one_to_one')
    affiliated_frame = plan['tail_attached_df'].drop(
        columns=['is_contaminated', 'is_gt_tail'], errors='ignore'
    ).merge(
        labels[analysis_columns], on='sample_idx', how='left', validate='one_to_one'
    )
    if open_frame[['is_contaminated', 'is_gt_tail']].isna().any().any():
        raise ValueError('{} {} open branch lacks analysis labels'.format(scenario, geometry_name))
    if affiliated_frame[['is_contaminated', 'is_gt_tail']].isna().any().any():
        raise ValueError('{} {} affiliated branch lacks analysis labels'.format(scenario, geometry_name))
    open_comp = _composition(open_frame)
    affiliated_comp = _composition(affiliated_frame)
    total_clean_tail = open_comp['clean_gt_tail'] + affiliated_comp['clean_gt_tail']
    open_precision = (
        open_comp['clean_gt_tail'] / open_comp['num_samples'] if open_comp['num_samples'] else None
    )
    open_recall = open_comp['clean_gt_tail'] / total_clean_tail if total_clean_tail else None
    rgd_summary = plan['rgd_split_summary']
    return {
        'scenario': scenario,
        'geometry': geometry_name,
        'split_status': rgd_summary['status'],
        'split_valid': bool(rgd_summary['split_valid']),
        'split_threshold': rgd_summary['threshold'],
        'num_valid_groups': int(rgd_summary['num_valid_groups']),
        'num_T0': int(open_comp['num_samples'] + affiliated_comp['num_samples']),
        'num_T_open': open_comp['num_samples'],
        'num_T_aff': affiliated_comp['num_samples'],
        'T_open_clean_gt_tail': open_comp['clean_gt_tail'],
        'T_open_clean_gt_head': open_comp['clean_gt_head'],
        'T_open_contamination': open_comp['contamination'],
        'T_aff_clean_gt_tail': affiliated_comp['clean_gt_tail'],
        'T_aff_clean_gt_head': affiliated_comp['clean_gt_head'],
        'T_aff_contamination': affiliated_comp['contamination'],
        'open_clean_gt_tail_precision': open_precision,
        'open_clean_gt_tail_recall': open_recall,
    }, open_frame, affiliated_frame


def _replay_scenario(results_root, output_root, scenario):
    run_root = os.path.join(results_root, '{}_full'.format(scenario), 'tailguard')
    paths = {
        'embeddings': os.path.join(run_root, 'prepare', 'grouping_embeddings.pt'),
        'details': os.path.join(
            run_root, 'prepare', 'tail_sampler_analysis_only', 'sampler_analysis_details.csv'
        ),
        'head_decisions': os.path.join(run_root, 'stage2', 'h_prune_decisions.csv'),
        'recorded_open': os.path.join(run_root, 'attachment', 'tail_open_samples.csv'),
        'recorded_affiliated': os.path.join(run_root, 'attachment', 'tail_attached_samples.csv'),
    }
    embeddings = _load_embeddings(paths['embeddings'])
    details = _read_csv(paths['details'])
    labels = _analysis_labels(details)
    tail = details.loc[details['tail_candidate'].astype(int) == 1].copy().reset_index(drop=True)
    raw_head, clean_head = _head_frames(_read_csv(paths['head_decisions']))
    args = SimpleNamespace(
        tg_min_clean_group_size=3,
        tg_rgd_split_mode='segmented_bic',
        tg_elbow_min_segment=3,
    )
    scenario_dir = os.path.join(output_root, scenario)
    os.makedirs(scenario_dir, exist_ok=True)
    rows = []
    branch_ids = {}
    for geometry_name, head_frame in zip(BRANCHES, (raw_head, clean_head)):
        plan = build_tail_attachment_plan(tail, head_frame, embeddings, args, 'rgd')
        row, open_frame, affiliated_frame = _branch_summary(
            scenario, geometry_name, plan, labels
        )
        rows.append(row)
        branch_ids[geometry_name] = {
            'open': set(open_frame['sample_idx'].astype(int)),
            'affiliated': set(affiliated_frame['sample_idx'].astype(int)),
        }
        open_frame.to_csv(
            os.path.join(scenario_dir, '{}_open.csv'.format(geometry_name)), index=False
        )
        affiliated_frame.to_csv(
            os.path.join(scenario_dir, '{}_affiliated.csv'.format(geometry_name)), index=False
        )
        plan['rgd_scores_df'].to_csv(
            os.path.join(scenario_dir, '{}_rgd_scores.csv'.format(geometry_name)), index=False
        )
        plan['rgd_bic_candidates_df'].to_csv(
            os.path.join(scenario_dir, '{}_rgd_bic_candidates.csv'.format(geometry_name)), index=False
        )
        _write_json(
            os.path.join(scenario_dir, '{}_rgd_summary.json'.format(geometry_name)),
            plan['rgd_split_summary'],
        )

    raw_open = branch_ids['unpurified_h0']['open']
    clean_open = branch_ids['purified_hc']['open']
    transition = {
        'scenario': scenario,
        'num_open_in_both': len(raw_open & clean_open),
        'num_open_to_affiliated_after_H': len(raw_open - clean_open),
        'num_affiliated_to_open_after_H': len(clean_open - raw_open),
        'num_role_changes': len(raw_open ^ clean_open),
    }

    recorded_open = set(_read_csv(paths['recorded_open'])['sample_idx'].astype(int))
    recorded_affiliated = set(_read_csv(paths['recorded_affiliated'])['sample_idx'].astype(int))
    if recorded_open != branch_ids['purified_hc']['open']:
        raise RuntimeError('{} replayed Hc open branch differs from recorded artifacts'.format(scenario))
    if recorded_affiliated != branch_ids['purified_hc']['affiliated']:
        raise RuntimeError('{} replayed Hc affiliated branch differs from recorded artifacts'.format(scenario))

    provenance = {
        'schema_version': 1,
        'scenario': scenario,
        'no_training': True,
        'analysis_only': True,
        'fixed_inputs': {
            key: {'path': os.path.realpath(path), 'sha256': _sha256(path)}
            for key, path in paths.items()
        },
        'fixed_protocol': {
            'membership_mode': 'rgd',
            'split_mode': 'segmented_bic',
            'min_segment': 3,
            'min_clean_group_size': 3,
        },
        'counterfactual': 'same T0/group assignments/embeddings; supported geometry includes or excludes H removals',
        'purified_replay_matches_recorded_artifacts': True,
    }
    pd.DataFrame(rows).to_csv(os.path.join(scenario_dir, 'geometry_comparison.csv'), index=False)
    _write_json(os.path.join(scenario_dir, 'transition_summary.json'), transition)
    _write_json(os.path.join(scenario_dir, 'replay_provenance.json'), provenance)
    return rows, transition


def _sum(rows, key):
    return sum(row[key] for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--results_root',
        default='/home/linux/projects/results/tailguard/dependency_ablation_v5',
    )
    parser.add_argument('--output_dir', default='analysis_outputs/rgd_geometry_replay_v1')
    args = parser.parse_args()
    output_root = os.path.realpath(args.output_dir)
    os.makedirs(output_root, exist_ok=True)
    all_rows, transitions = [], []
    for scenario in SCENARIOS:
        rows, transition = _replay_scenario(
            os.path.realpath(args.results_root), output_root, scenario
        )
        all_rows.extend(rows)
        transitions.append(transition)

    fields = list(all_rows[0])
    with open(os.path.join(output_root, 'geometry_comparison_by_scenario.csv'), 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    with open(os.path.join(output_root, 'role_transitions_by_scenario.csv'), 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(transitions[0]))
        writer.writeheader()
        writer.writerows(transitions)

    aggregate = {}
    for branch in BRANCHES:
        branch_rows = [row for row in all_rows if row['geometry'] == branch]
        num_open = _sum(branch_rows, 'num_T_open')
        num_open_tail = _sum(branch_rows, 'T_open_clean_gt_tail')
        total_clean_tail = num_open_tail + _sum(branch_rows, 'T_aff_clean_gt_tail')
        aggregate[branch] = {
            'num_T0': _sum(branch_rows, 'num_T0'),
            'num_T_open': num_open,
            'num_T_aff': _sum(branch_rows, 'num_T_aff'),
            'T_open_clean_gt_tail': num_open_tail,
            'T_open_clean_gt_head': _sum(branch_rows, 'T_open_clean_gt_head'),
            'T_open_contamination': _sum(branch_rows, 'T_open_contamination'),
            'T_aff_clean_gt_tail': _sum(branch_rows, 'T_aff_clean_gt_tail'),
            'T_aff_clean_gt_head': _sum(branch_rows, 'T_aff_clean_gt_head'),
            'T_aff_contamination': _sum(branch_rows, 'T_aff_contamination'),
            'open_clean_gt_tail_precision_micro': num_open_tail / num_open if num_open else None,
            'open_clean_gt_tail_recall_micro': num_open_tail / total_clean_tail if total_clean_tail else None,
            'num_valid_splits': sum(1 for row in branch_rows if row['split_valid']),
        }
    summary = {
        'schema_version': 1,
        'no_training': True,
        'analysis_only': True,
        'num_scenarios': len(SCENARIOS),
        'aggregate': aggregate,
        'role_transitions': {
            'num_open_in_both': _sum(transitions, 'num_open_in_both'),
            'num_open_to_affiliated_after_H': _sum(transitions, 'num_open_to_affiliated_after_H'),
            'num_affiliated_to_open_after_H': _sum(transitions, 'num_affiliated_to_open_after_H'),
            'num_role_changes': _sum(transitions, 'num_role_changes'),
        },
        'claim_boundary': 'Post-hoc structural replay; oracle labels are used only to audit composition.',
    }
    _write_json(os.path.join(output_root, 'rgd_geometry_replay_summary.json'), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
