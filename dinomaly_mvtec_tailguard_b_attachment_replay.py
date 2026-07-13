import argparse
import json
import os
from types import SimpleNamespace

import pandas as pd
import torch

from tailguard_b_artifacts import (
    save_tailguard_b_attachment_artifacts,
    save_tailguard_b_stage2_artifacts,
)
from tailguard_b_attachment import build_tail_attachment_plan
from tailguard_b_gbps import (
    build_tailguard_b_head_delete_plan,
    build_tailguard_b_stage2_plan,
    classify_tail_attached_stable_risk,
)


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def _load_torch(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


REPLAY_DEFAULTS = {
    'tgb_attachment_membership_mode': 'empirical_conformity',
    'tgb_min_clean_group_size': 3,
    'tgb_elbow_min_segment': 3,
    'tgb_stable_window': 3,
    'tgb_stable_min_observations': 1,
    'tgb_t_attached_noise_p_high_thr': 0.5,
    'gbps_prune_ratio': 0.1,
    'gbps_min_keep_per_group': 20,
}


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
    resolved = {}
    for name, default in REPLAY_DEFAULTS.items():
        cli_value = getattr(parsed_args, name)
        resolved[name] = cli_value if cli_value is not None else source_config.get(name, default)
    status = 'source_config' if source_config else 'cli_or_defaults_without_source_config'
    return SimpleNamespace(**resolved), status, config_path if source_config else None


def replay(parsed_args):
    source_dir = os.path.realpath(parsed_args.source_tailguard_b_dir)
    output_dir = os.path.realpath(parsed_args.output_dir)
    if not os.path.isdir(source_dir):
        raise FileNotFoundError('source TailGuard-B directory does not exist: {}'.format(source_dir))
    if os.path.commonpath([source_dir, output_dir]) == source_dir:
        raise ValueError('output_dir must be outside source_tailguard_b_dir to protect source artifacts')
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError('output_dir must be empty: {}'.format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    prepare_dir = os.path.join(source_dir, 'prepare')
    gbps_dir = os.path.join(source_dir, 'gbps')
    metadata_df = pd.read_csv(os.path.join(prepare_dir, 'tailguard_train_metadata.csv'))
    head_group_assignments_df = pd.read_csv(os.path.join(prepare_dir, 'head_group_assignments.csv'))
    grouping_embeddings_payload = _load_torch(os.path.join(prepare_dir, 'grouping_embeddings.pt'))
    selected_iter, score_source_iter = _resolve_selected_score_source(source_dir, parsed_args.gbps_selected_iter)
    selected_scores_path = os.path.join(gbps_dir, 'iter_{:05d}'.format(score_source_iter), 'train_scores.csv')
    if not os.path.isfile(selected_scores_path):
        raise FileNotFoundError('selected GBPS score artifact does not exist: {}'.format(selected_scores_path))
    selected_scores_df = pd.read_csv(selected_scores_path)
    replay_args, config_status, config_path = _resolve_replay_config(source_dir, parsed_args)
    if config_status != 'source_config':
        print('warning: source run has no attachment replay config; using explicit CLI values or documented defaults')

    head_delete_plan = build_tailguard_b_head_delete_plan(
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
        replay_args.tgb_attachment_membership_mode,
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

    tail_head_normal_df, tail_head_noise_df, stable_risk_df = classify_tail_attached_stable_risk(
        tail_attached_df,
        gbps_dir,
        selected_iter,
        replay_args,
    )
    attachment_dir = os.path.join(output_dir, 'attachment')
    attachment_artifacts = save_tailguard_b_attachment_artifacts(
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
    )
    stable_risk_path = os.path.join(attachment_dir, 'tail_attached_stable_risk.csv')
    stable_risk_df.to_csv(stable_risk_path, index=False)

    stage2_plan = build_tailguard_b_stage2_plan(
        head_delete_plan['h_clean_samples'],
        head_delete_plan['h_removed_samples'],
        tail_open_df,
        tail_head_normal_df,
        tail_head_noise_df,
        all_samples_df=head_delete_plan['selected_scores_df'],
    )
    stage2_artifacts = save_tailguard_b_stage2_artifacts(
        os.path.join(output_dir, 'stage2'),
        stage2_plan['retained_samples'],
        stage2_plan['removed_samples'],
        stage2_plan['summary'],
    )
    summary = {
        'source_tailguard_b_dir': source_dir,
        'selected_iter': int(selected_iter),
        'score_source_iter': int(score_source_iter),
        'attachment_membership_mode': replay_args.tgb_attachment_membership_mode,
        'replay_config_status': config_status,
        'source_replay_config_path': config_path,
        'attachment_summary': attachment_summary,
        'head_delete_summary': head_delete_plan['summary'],
        'stage2_summary': stage2_plan['summary'],
        'attachment_artifacts': attachment_artifacts,
        'tail_attached_stable_risk_csv': stable_risk_path,
        'stage2_artifacts': stage2_artifacts,
    }
    summary_path = os.path.join(output_dir, 'attachment_replay_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay TailGuard-B attachment from saved artifacts without model training')
    parser.add_argument('--source_tailguard_b_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gbps_selected_iter', type=int, default=None)
    parser.add_argument(
        '--tgb_attachment_membership_mode',
        type=str,
        default=None,
        choices=['empirical_conformity', 'h_calibrated'],
    )
    parser.add_argument('--tgb_min_clean_group_size', type=int, default=None)
    parser.add_argument('--tgb_elbow_min_segment', type=int, default=None)
    parser.add_argument('--tgb_stable_window', type=int, default=None)
    parser.add_argument('--tgb_stable_min_observations', type=int, default=None)
    parser.add_argument('--tgb_t_attached_noise_p_high_thr', type=float, default=None)
    parser.add_argument('--gbps_prune_ratio', type=float, default=None)
    parser.add_argument('--gbps_min_keep_per_group', type=int, default=None)
    replay(parser.parse_args())
