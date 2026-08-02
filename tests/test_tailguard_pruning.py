import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dinomaly_mvtec_tailguard_attachment_replay import (
    REPLAY_CONFIG_KEYS,
    _resolve_replay_config,
    _resolve_selected_score_source,
)
from gbps_runner_utils import summarize_contamination_labels
from tailguard_defaults import TAILGUARD_FINAL_DEFAULTS
from tailguard_diagnostics import build_tailguard_denoising_diagnostics
from tailguard_gbps import build_tailguard_head_delete_plan, build_tailguard_stage2_plan
from tailguard_pseudoclasses import build_tailguard_pseudoclass_registry


def _args(mode='adaptive', **overrides):
    values = {
        'gbps_prune_mode': mode,
        'gbps_prune_max_ratio': 0.10,
        'gbps_prune_stable_window': 3,
        'gbps_prune_stable_min_observations': 1,
        'gbps_prune_min_active_ratio': 0.5,
        'gbps_prune_ratio': 0.10,
        'gbps_min_keep_per_group': 20,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _dataset(group_sizes):
    score_rows = []
    metadata_rows = []
    assignment_rows = []
    sample_idx = 0
    for group_id, group_size in enumerate(group_sizes):
        for offset in range(group_size):
            image_score = float(offset + 1) / float(group_size + 1) + float(group_id)
            score_rows.append({
                'sample_idx': sample_idx,
                'class_id': 0,
                'base_idx': sample_idx,
                'image_score': image_score,
            })
            metadata_rows.append({
                'sample_idx': sample_idx,
                'sample_key': sample_idx,
                'img_path': 'group_{}/{}.png'.format(group_id, offset),
                'base_idx': sample_idx,
                'tail_candidate': 0,
                'pred_class_size': float(group_size),
                'tail_score': 0.0,
                'adaptive_angle': 5.0,
                'group_id': group_id,
                'group_size': group_size,
            })
            assignment_rows.append({'sample_key': sample_idx, 'group_id': group_id})
            sample_idx += 1
    return pd.DataFrame(score_rows), pd.DataFrame(metadata_rows), pd.DataFrame(assignment_rows)


def _write_metrics(gbps_dir, iter_number, group_sizes, pi_values, active_values):
    iter_dir = os.path.join(gbps_dir, 'iter_{:05d}'.format(iter_number))
    os.makedirs(iter_dir, exist_ok=True)
    rows = []
    for group_id, group_size in enumerate(group_sizes):
        is_active = bool(active_values[group_id])
        rows.append({
            'group_id': group_id,
            'group_size': group_size,
            'valid_group': True,
            'pi_high': float(pi_values[group_id]),
            'is_active_group': is_active,
            'q_g': 1.0 if is_active else 0.0,
            'delta_bic': 10.0,
            'sep': 1.0,
            'conf': 0.8,
        })
    pd.DataFrame(rows).to_csv(os.path.join(iter_dir, 'h_group_metrics.csv'), index=False)


class TailGuardPruningTest(unittest.TestCase):
    def test_defaults_select_adaptive_with_ten_percent_cap(self):
        self.assertEqual(TAILGUARD_FINAL_DEFAULTS['gbps_prune_mode'], 'adaptive')
        self.assertEqual(TAILGUARD_FINAL_DEFAULTS['gbps_prune_max_ratio'], 0.10)
        self.assertEqual(TAILGUARD_FINAL_DEFAULTS['gbps_prune_ratio'], 0.10)

    def test_unlabeled_prune_artifact_counts_all_rows(self):
        counts = summarize_contamination_labels(pd.DataFrame({'sample_idx': [1, 2, 3]}))
        self.assertEqual(counts, {
            'num_samples': 3,
            'num_labeled_clean': 0,
            'num_labeled_contaminated': 0,
            'num_unlabeled_samples': 3,
        })

    def test_fixed_mode_preserves_groupwise_top_ratio_without_metrics(self):
        scores, metadata, assignments = _dataset([100, 21, 20])
        plan = build_tailguard_head_delete_plan(
            scores,
            assignments,
            metadata,
            _args(mode='fixed'),
        )
        counts = plan['h_removed_samples'].groupby('group_id').size().to_dict()
        self.assertEqual(counts, {0: 10, 1: 1})
        self.assertEqual(plan['summary']['num_h_removed'], 11)
        self.assertEqual(plan['summary']['prune_mode'], 'fixed')

    def test_adaptive_mode_uses_estimated_ratio_below_cap(self):
        group_sizes = [100]
        scores, metadata, assignments = _dataset(group_sizes)
        with tempfile.TemporaryDirectory() as gbps_dir:
            _write_metrics(gbps_dir, 10, group_sizes, [0.05], [True])
            plan = build_tailguard_head_delete_plan(
                scores,
                assignments,
                metadata,
                _args(),
                gbps_dir=gbps_dir,
                selected_iter=10,
            )
        self.assertEqual(plan['summary']['num_h_removed'], 5)
        row = plan['h_prune_group_summary_df'].iloc[0]
        self.assertEqual(float(row['adaptive_requested_ratio']), 0.05)
        self.assertFalse(bool(row['cap_applied']))
        self.assertTrue(plan['h_prune_decisions_df']['group_size'].eq(100).all())
        self.assertTrue(plan['h_clean_samples']['group_size'].eq(100).all())
        self.assertTrue(plan['h_removed_samples']['group_size'].eq(100).all())

    def test_adaptive_mode_applies_cap_stability_selected_gate_and_min_keep(self):
        group_sizes = [100, 100, 21, 20, 100]
        pi_values = [0.40, 0.05, 0.40, 0.40, 0.40]
        scores, metadata, assignments = _dataset(group_sizes)
        with tempfile.TemporaryDirectory() as gbps_dir:
            _write_metrics(gbps_dir, 10, group_sizes, pi_values, [True, False, True, True, True])
            _write_metrics(gbps_dir, 20, group_sizes, pi_values, [False, False, True, True, True])
            _write_metrics(gbps_dir, 30, group_sizes, pi_values, [True, True, True, True, False])
            plan = build_tailguard_head_delete_plan(
                scores,
                assignments,
                metadata,
                _args(),
                gbps_dir=gbps_dir,
                selected_iter=30,
            )

        counts = plan['h_removed_samples'].groupby('group_id').size().to_dict()
        self.assertEqual(counts, {0: 10, 2: 1})
        group_summary = plan['h_prune_group_summary_df'].set_index('group_id')
        self.assertEqual(int(group_summary.loc[0, 'removed']), 10)
        self.assertTrue(bool(group_summary.loc[0, 'cap_applied']))
        self.assertEqual(int(group_summary.loc[1, 'removed']), 0)
        self.assertEqual(group_summary.loc[1, 'decision_reason'], 'unstable_active_gate')
        self.assertEqual(int(group_summary.loc[2, 'removed']), 1)
        self.assertEqual(group_summary.loc[2, 'decision_reason'], 'min_keep_cap_applied')
        self.assertEqual(int(group_summary.loc[3, 'removed']), 0)
        self.assertEqual(int(group_summary.loc[4, 'removed']), 0)
        self.assertEqual(group_summary.loc[4, 'decision_reason'], 'selected_group_inactive')

    def test_exactly_half_active_is_not_stable(self):
        group_sizes = [100]
        scores, metadata, assignments = _dataset(group_sizes)
        with tempfile.TemporaryDirectory() as gbps_dir:
            _write_metrics(gbps_dir, 10, group_sizes, [0.40], [False])
            _write_metrics(gbps_dir, 20, group_sizes, [0.40], [True])
            plan = build_tailguard_head_delete_plan(
                scores,
                assignments,
                metadata,
                _args(gbps_prune_stable_window=2),
                gbps_dir=gbps_dir,
                selected_iter=20,
            )
        self.assertEqual(plan['summary']['num_h_removed'], 0)
        row = plan['h_prune_group_summary_df'].iloc[0]
        self.assertEqual(float(row['stable_active_ratio']), 0.5)
        self.assertFalse(bool(row['stable_group_active']))

    def test_zero_removal_stage2_preserves_schema_for_diagnostics(self):
        """A0's no-cleanup path must still expose an empty sample-indexed set."""
        initial = pd.DataFrame({
            'sample_idx': [0, 1],
            'sample_key': [0, 1],
            'img_path': ['toy/0.png', 'toy/1.png'],
            'base_idx': [0, 1],
            'class_id': [0, 0],
            'class_name': ['toy', 'toy'],
            'is_contaminated': [0, 0],
            'group_id': [0, 0],
        })
        empty_h = initial.iloc[0:0].copy()
        empty_tail_open = initial.iloc[0:0].copy()
        empty_tail_open['adaptive_angle'] = pd.Series(dtype=float)
        empty_tail_attached = initial.iloc[0:0].copy()
        empty_tail_attached['best_group_id'] = pd.Series(dtype=int)

        stage2_plan = build_tailguard_stage2_plan(
            initial,
            empty_h,
            empty_tail_open,
            empty_tail_attached,
            empty_tail_attached,
            all_samples_df=initial,
        )

        self.assertEqual(len(stage2_plan['removed_samples']), 0)
        self.assertIn('sample_idx', stage2_plan['removed_samples'].columns)
        summary, by_class = build_tailguard_denoising_diagnostics(
            initial,
            stage2_plan['retained_samples'],
            stage2_plan['removed_samples'],
            h_removed_samples_df=empty_h,
            tail_head_noise_samples_df=empty_tail_attached,
        )
        self.assertEqual(summary['cleanup_status'], 'completed')
        self.assertEqual(summary['decision_counts']['num_removed'], 0)
        self.assertEqual(summary['decision_counts']['num_retained'], 2)
        self.assertEqual(len(by_class), 1)

        registry = build_tailguard_pseudoclass_registry(
            initial,
            empty_tail_attached,
            empty_tail_open,
            stage2_plan['retained_samples'],
            stage2_plan['removed_samples'],
            {
                'sample_idx': [0, 1],
                'embeddings': torch.eye(2),
                'embedding_source': 'test',
            },
        )
        self.assertEqual(registry['summary']['status'], 'completed')
        self.assertEqual(registry['summary']['num_removed_samples'], 0)

    def test_replay_schema_preserves_legacy_fixed_semantics(self):
        parsed = SimpleNamespace(**{name: None for name in REPLAY_CONFIG_KEYS})
        with tempfile.TemporaryDirectory() as source_dir:
            config_path = os.path.join(source_dir, 'attachment_replay_config.json')
            with open(config_path, 'w', encoding='utf-8') as file:
                json.dump({
                    'schema_version': 2,
                    'method_name': 'tailguard',
                    'config_profile': 'final',
                    'gbps_prune_ratio': 0.1,
                }, file)
            resolved, status, _ = _resolve_replay_config(source_dir, parsed)
            self.assertEqual(status, 'canonical_legacy_schema_fixed_pruning')
            self.assertEqual(resolved.gbps_prune_mode, 'fixed')

            with open(config_path, 'w', encoding='utf-8') as file:
                json.dump({
                    'schema_version': 3,
                    'method_name': 'tailguard',
                    'config_profile': 'final',
                    'gbps_prune_mode': 'adaptive',
                    'gbps_prune_max_ratio': 0.1,
                }, file)
            resolved, status, _ = _resolve_replay_config(source_dir, parsed)
            self.assertEqual(status, 'canonical_current_schema')
            self.assertEqual(resolved.gbps_prune_mode, 'adaptive')

    def test_replay_score_source_uses_resolved_gbps_directory_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as gbps_dir:
            summary_path = os.path.join(gbps_dir, 'gbps_trigger_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as file:
                json.dump({'gbps_selected_iter': 30, 'score_source_iter': 30}, file)
            self.assertEqual(_resolve_selected_score_source(gbps_dir, None), (30, 30))

            with open(summary_path, 'w', encoding='utf-8') as file:
                json.dump({'gbps_selected_iter': 30, 'score_source_iter': 20}, file)
            with self.assertRaisesRegex(ValueError, 'iteration mismatch'):
                _resolve_selected_score_source(gbps_dir, None)


if __name__ == '__main__':
    unittest.main()
