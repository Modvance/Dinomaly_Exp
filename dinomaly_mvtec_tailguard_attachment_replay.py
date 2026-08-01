"""Replay TailGuard attachment and pseudo-class construction from saved artifacts."""

import argparse
import json
import os
from types import SimpleNamespace

import pandas as pd
import torch

from tailguard_artifacts import (
    save_tailguard_attachment_artifacts,
    save_tailguard_denoising_diagnostics,
    save_tailguard_pseudoclass_artifacts,
    save_tailguard_pseudoclass_report_artifacts,
    save_tailguard_stage2_artifacts,
)
from tailguard_attachment import build_tail_attachment_plan
from tailguard_diagnostics import (
    build_tailguard_denoising_diagnostics,
    build_tailguard_pseudoclass_report,
)
from tailguard_gbps import (
    build_tailguard_head_delete_plan,
    build_tailguard_stage2_plan,
    classify_tail_attached_stable_risk,
)
from tailguard_pseudoclasses import build_tailguard_pseudoclass_registry
from tailguard_defaults import (
    TAILGUARD_CONFIG_PROFILE,
    TAILGUARD_CONFIG_SCHEMA_VERSION,
    TAILGUARD_FINAL_DEFAULTS,
    TAILGUARD_METHOD_NAME,
)


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def _load_torch(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


# An artifact without a versioned replay config belongs to the pre-canonical
# TailGuard-B implementation.  Keep its historical fallback semantics so a
# paper result is never silently reinterpreted with the new final profile.
LEGACY_REPLAY_DEFAULTS = {
    'tg_attachment_membership_mode': 'empirical_conformity',
    'tg_min_clean_group_size': 3,
    'tg_elbow_min_segment': 3,
    'tg_rgd_split_mode': 'largest_positive_gap',
    'tg_stable_window': 3,
    'tg_stable_min_observations': 1,
    'tg_t_attached_noise_p_high_thr': 0.5,
    'gbps_prune_ratio': 0.1,
    'gbps_min_keep_per_group': 20,
}

REPLAY_CONFIG_KEYS = tuple(LEGACY_REPLAY_DEFAULTS)


def _source_config_value(source_config, name, default):
    if name in source_config:
        return source_config[name]
    if name.startswith('tg_'):
        legacy_name = 'tgb_' + name[len('tg_'):]
        if legacy_name in source_config:
            return source_config[legacy_name]
    return default


def _is_canonical_source_config(source_config):
    return (
        source_config.get('method_name') == TAILGUARD_METHOD_NAME
        and source_config.get('config_profile') == TAILGUARD_CONFIG_PROFILE
        and int(source_config.get('schema_version', 0)) >= TAILGUARD_CONFIG_SCHEMA_VERSION
    )


def _resolve_selected_score_source(source_dir, requested_iter):
    if requested_iter is not None:
        selected_iter = int(requested_iter)
        return selected_iter, selected_iter
    trigger_path = os.path.join(source_dir, 'gbps', 'gbps_trigger_summary.json')
    trigger_summary = _load_json(trigger_path)
    selected_iter = trigger_summary.get('gbps_selected_iter')
    if selected_iter is None:
        raise ValueError('GBPS trigger summary has no selected iteration: {}'.format(trigger_path))
    selected_iter = int(selected_iter)
    return selected_iter, int(trigger_summary.get('score_source_iter', selected_iter))


def _resolve_replay_config(source_dir, parsed_args):
    config_path = os.path.join(source_dir, 'attachment_replay_config.json')
    source_config = _load_json(config_path) if os.path.isfile(config_path) else {}
    is_canonical = _is_canonical_source_config(source_config)
    defaults = TAILGUARD_FINAL_DEFAULTS if is_canonical else LEGACY_REPLAY_DEFAULTS
    resolved = {}
    for name in REPLAY_CONFIG_KEYS:
        cli_value = getattr(parsed_args, name, None)
        resolved[name] = (
            cli_value
            if cli_value is not None
            else _source_config_value(source_config, name, defaults[name])
        )
    if is_canonical:
        status = 'canonical_source_config'
    elif source_config:
        status = 'legacy_source_config'
    else:
        status = 'legacy_defaults_without_source_config'
    return SimpleNamespace(**resolved), status, config_path if source_config else None


def replay(parsed_args):
    source_dir = os.path.realpath(parsed_args.source_tailguard_dir)
    output_dir = os.path.realpath(parsed_args.output_dir)
    if not os.path.isdir(source_dir):
        raise FileNotFoundError('source TailGuard directory does not exist: {}'.format(source_dir))
    if os.path.commonpath([source_dir, output_dir]) == source_dir:
        raise ValueError('output_dir must be outside source_tailguard_dir to protect source artifacts')
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError('output_dir must be empty: {}'.format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    prepare_dir = os.path.join(source_dir, 'prepare')
    source_replay_config_path = os.path.join(source_dir, 'attachment_replay_config.json')
    source_replay_config = _load_json(source_replay_config_path) if os.path.isfile(source_replay_config_path) else {}
    configured_gbps_dir = _source_config_value(source_replay_config, 'tg_gbps_dir', None)
    if configured_gbps_dir is not None:
        gbps_dir = os.path.realpath(configured_gbps_dir)
    else:
        gbps_dir_relpath = _source_config_value(source_replay_config, 'tg_gbps_dir_relpath', 'gbps')
        gbps_dir = os.path.realpath(os.path.join(source_dir, gbps_dir_relpath))
    metadata_df = pd.read_csv(os.path.join(prepare_dir, 'tailguard_train_metadata.csv'))
    head_group_assignments_df = pd.read_csv(os.path.join(prepare_dir, 'head_group_assignments.csv'))
    grouping_embeddings_payload = _load_torch(os.path.join(prepare_dir, 'grouping_embeddings.pt'))
    cls_embeddings_path = os.path.join(prepare_dir, 'cls_embeddings.pt')
    cls_embeddings_payload = _load_torch(cls_embeddings_path) if os.path.isfile(cls_embeddings_path) else None
    selected_iter, score_source_iter = _resolve_selected_score_source(source_dir, parsed_args.gbps_selected_iter)
    selected_scores_path = os.path.join(gbps_dir, 'iter_{:05d}'.format(score_source_iter), 'train_scores.csv')
    if not os.path.isfile(selected_scores_path):
        raise FileNotFoundError('selected GBPS score artifact does not exist: {}'.format(selected_scores_path))
    selected_scores_df = pd.read_csv(selected_scores_path)
    analysis_metadata_path = os.path.join(prepare_dir, 'tailguard_train_analysis_metadata.csv')
    if os.path.isfile(analysis_metadata_path):
        analysis_metadata_df = pd.read_csv(analysis_metadata_path)
        analysis_metadata_source = analysis_metadata_path
    else:
        analysis_metadata_df = selected_scores_df.copy()
        analysis_metadata_source = selected_scores_path
    prepare_summary_path = os.path.join(prepare_dir, 'prepare_summary.json')
    prepare_artifact_summary = _load_json(prepare_summary_path) if os.path.isfile(prepare_summary_path) else {}
    analysis_metadata_provenance = prepare_artifact_summary.get('metadata', {}).get('summary', {}).get(
        'analysis_metadata_provenance',
        {},
    )
    replay_args, config_status, config_path = _resolve_replay_config(source_dir, parsed_args)
    if config_status != 'canonical_source_config':
        print(
            'warning: replaying a legacy TailGuard-B artifact; explicit CLI values and '
            'legacy empirical-conformity/largest-gap fallbacks are used where metadata is absent'
        )

    head_delete_plan = build_tailguard_head_delete_plan(
        selected_scores_df,
        head_group_assignments_df,
        metadata_df,
        replay_args,
    )
    tail_df = head_delete_plan['selected_scores_df'].loc[
        head_delete_plan['selected_scores_df']['tail_candidate'].astype(int) == 1
    ].copy().reset_index(drop=True)
    attachment_plan = build_tail_attachment_plan(
        tail_df,
        head_delete_plan['h_clean_samples'],
        grouping_embeddings_payload,
        replay_args,
        replay_args.tg_attachment_membership_mode,
    )
    geometry = attachment_plan['geometry']
    conformity_df = attachment_plan['conformity_df']
    attachment_scores_df = attachment_plan['attachment_scores_df']
    tail_open_df = attachment_plan['tail_open_df']
    tail_attached_df = attachment_plan['tail_attached_df']
    attachment_summary = attachment_plan['attachment_summary']
    membership_scores_df = attachment_plan['membership_scores_df']
    membership_calibration_df = attachment_plan['membership_calibration_df']
    membership_summary = attachment_plan['membership_summary']
    rgd_distances_df = attachment_plan['rgd_distances_df']
    rgd_scores_df = attachment_plan['rgd_scores_df']
    rgd_split_summary = attachment_plan['rgd_split_summary']
    rgd_bic_candidates_df = attachment_plan['rgd_bic_candidates_df']

    tail_head_normal_df, tail_head_noise_df, stable_risk_df = classify_tail_attached_stable_risk(
        tail_attached_df,
        gbps_dir,
        selected_iter,
        replay_args,
    )
    attachment_dir = os.path.join(output_dir, 'attachment')
    attachment_artifacts = save_tailguard_attachment_artifacts(
        attachment_dir,
        geometry,
        conformity_df,
        attachment_scores_df,
        attachment_summary,
        tail_open_df,
        tail_attached_df,
        tail_head_normal_df,
        tail_head_noise_df,
        membership_scores_df=membership_scores_df,
        membership_calibration_df=membership_calibration_df,
        membership_summary=membership_summary,
        rgd_distances_df=rgd_distances_df,
        rgd_scores_df=rgd_scores_df,
        rgd_split_summary=rgd_split_summary,
        rgd_bic_candidates_df=rgd_bic_candidates_df,
    )
    stable_risk_path = os.path.join(attachment_dir, 'tail_attached_stable_risk.csv')
    stable_risk_df.to_csv(stable_risk_path, index=False)

    stage2_plan = build_tailguard_stage2_plan(
        head_delete_plan['h_clean_samples'],
        head_delete_plan['h_removed_samples'],
        tail_open_df,
        tail_head_normal_df,
        tail_head_noise_df,
        all_samples_df=head_delete_plan['selected_scores_df'],
    )
    stage2_artifacts = save_tailguard_stage2_artifacts(
        os.path.join(output_dir, 'stage2'),
        stage2_plan['retained_samples'],
        stage2_plan['removed_samples'],
        stage2_plan['summary'],
    )
    denoising_diagnostics_summary, denoising_diagnostics_by_class = build_tailguard_denoising_diagnostics(
        analysis_metadata_df,
        stage2_plan['retained_samples'],
        stage2_plan['removed_samples'],
        h_removed_samples_df=head_delete_plan['h_removed_samples'],
        tail_head_noise_samples_df=tail_head_noise_df,
        manifest_path=analysis_metadata_provenance.get('manifest_path'),
        manifest_requested_path=analysis_metadata_provenance.get('manifest_requested_path'),
        label_source=analysis_metadata_source,
    )
    denoising_diagnostics_artifacts = save_tailguard_denoising_diagnostics(
        output_dir,
        denoising_diagnostics_summary,
        denoising_diagnostics_by_class,
    )
    pseudoclass_registry = None
    pseudoclass_artifacts = None
    pseudoclass_report_artifacts = None
    if cls_embeddings_payload is None:
        pseudoclass_status = 'not_available_missing_cls_embeddings'
        pseudoclass_reason = 'prepare/cls_embeddings.pt is unavailable'
    else:
        pseudoclass_registry = build_tailguard_pseudoclass_registry(
            head_delete_plan['h_clean_samples'],
            tail_head_normal_df,
            tail_open_df,
            stage2_plan['retained_samples'],
            stage2_plan['removed_samples'],
            cls_embeddings_payload,
        )
        pseudoclass_artifacts = save_tailguard_pseudoclass_artifacts(
            os.path.join(output_dir, 'pseudoclasses'),
            pseudoclass_registry['members_df'],
            pseudoclass_registry['classes_df'],
            pseudoclass_registry['tail_edges_df'],
            pseudoclass_registry['summary'],
        )
        pseudoclass_report, pseudoclass_predictions, pseudoclass_summary_df, pseudoclass_contingency_df = (
            build_tailguard_pseudoclass_report(
                pseudoclass_registry['members_df'],
                pseudoclass_registry['classes_df'],
                cls_embeddings_payload,
                analysis_metadata_df,
            )
        )
        pseudoclass_report_artifacts = save_tailguard_pseudoclass_report_artifacts(
            output_dir,
            pseudoclass_report,
            pseudoclass_predictions,
            pseudoclass_summary_df,
            pseudoclass_contingency_df,
        )
        pseudoclass_status = 'completed'
        pseudoclass_reason = None
    summary = {
        'source_tailguard_dir': source_dir,
        'source_gbps_dir': gbps_dir,
        'selected_iter': int(selected_iter),
        'score_source_iter': int(score_source_iter),
        'attachment_membership_mode': replay_args.tg_attachment_membership_mode,
        'rgd_split_mode': replay_args.tg_rgd_split_mode,
        'replay_config_status': config_status,
        'source_replay_config_path': config_path,
        'attachment_summary': attachment_summary,
        'head_delete_summary': head_delete_plan['summary'],
        'stage2_summary': stage2_plan['summary'],
        'attachment_artifacts': attachment_artifacts,
        'tail_attached_stable_risk_csv': stable_risk_path,
        'stage2_artifacts': stage2_artifacts,
        'pseudo_class_status': pseudoclass_status,
        'pseudo_class_reason': pseudoclass_reason,
        'pseudo_class_artifacts': pseudoclass_artifacts,
        'pseudo_class_report_artifacts': pseudoclass_report_artifacts,
        'pseudo_class_summary': None if pseudoclass_registry is None else {
            'num_pseudo_classes': pseudoclass_registry['summary']['num_pseudo_classes'],
            'num_head_pseudo_classes': pseudoclass_registry['summary']['num_head_pseudo_classes'],
            'num_tail_pseudo_classes': pseudoclass_registry['summary']['num_tail_pseudo_classes'],
        },
        'final_memory_rebuilt': False,
        'denoising_diagnostics_artifacts': denoising_diagnostics_artifacts,
        'denoising_diagnostics_summary': {
            'cleanup_status': denoising_diagnostics_summary['cleanup_status'],
            'decision_counts': denoising_diagnostics_summary['decision_counts'],
            'contamination_counts': denoising_diagnostics_summary['contamination_counts'],
            'contamination_metrics': denoising_diagnostics_summary['contamination_metrics'],
        },
    }
    summary_path = os.path.join(output_dir, 'attachment_replay_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _add_hidden_legacy_alias(parser, option, dest, **kwargs):
    parser.add_argument(
        option,
        dest=dest,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
        **kwargs,
    )


def build_parser():
    parser = argparse.ArgumentParser(description='Replay TailGuard attachment from saved artifacts without model training')
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--source_tailguard_dir', dest='source_tailguard_dir', type=str)
    source_group.add_argument(
        '--source_tailguard_b_dir',
        dest='source_tailguard_dir',
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gbps_selected_iter', type=int, default=None)
    parser.add_argument(
        '--tg_attachment_membership_mode',
        type=str,
        default=None,
        choices=['empirical_conformity', 'h_calibrated', 'rgd'],
    )
    parser.add_argument('--tg_min_clean_group_size', type=int, default=None)
    parser.add_argument('--tg_elbow_min_segment', type=int, default=None)
    parser.add_argument(
        '--tg_rgd_split_mode',
        type=str,
        default=None,
        choices=['segmented_bic', 'largest_positive_gap'],
    )
    parser.add_argument('--tg_stable_window', type=int, default=None)
    parser.add_argument('--tg_stable_min_observations', type=int, default=None)
    parser.add_argument('--tg_t_attached_noise_p_high_thr', type=float, default=None)
    parser.add_argument('--gbps_prune_ratio', type=float, default=None)
    parser.add_argument('--gbps_min_keep_per_group', type=int, default=None)
    _add_hidden_legacy_alias(
        parser,
        '--tgb_attachment_membership_mode',
        'tg_attachment_membership_mode',
        type=str,
        choices=['empirical_conformity', 'h_calibrated', 'rgd'],
    )
    _add_hidden_legacy_alias(parser, '--tgb_min_clean_group_size', 'tg_min_clean_group_size', type=int)
    _add_hidden_legacy_alias(parser, '--tgb_elbow_min_segment', 'tg_elbow_min_segment', type=int)
    _add_hidden_legacy_alias(
        parser,
        '--tgb_rgd_split_mode',
        'tg_rgd_split_mode',
        type=str,
        choices=['segmented_bic', 'largest_positive_gap'],
    )
    _add_hidden_legacy_alias(parser, '--tgb_stable_window', 'tg_stable_window', type=int)
    _add_hidden_legacy_alias(parser, '--tgb_stable_min_observations', 'tg_stable_min_observations', type=int)
    _add_hidden_legacy_alias(
        parser,
        '--tgb_t_attached_noise_p_high_thr',
        'tg_t_attached_noise_p_high_thr',
        type=float,
    )
    return parser


if __name__ == '__main__':
    replay(build_parser().parse_args())
