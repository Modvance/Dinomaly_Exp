"""Run and aggregate the frozen 30-run TailGuard dual-reference protocol."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict

from sklearn.metrics import roc_auc_score


SCENARIOS = (
    ('mvtec', 'pareto'),
    ('mvtec', 'step_k4'),
    ('mvtec', 'step_k1'),
    ('visa', 'pareto'),
    ('visa', 'step_k4'),
    ('visa', 'step_k1'),
)
GUARD_REPLACEMENTS = {
    'visa_step_k4_seed02': 'visa_step_k4_seed02_full_guard',
    'visa_step_k1_seed03': 'visa_step_k1_seed03_full_guard',
    'visa_step_k1_seed05': 'visa_step_k1_seed05_full_guard',
}
REPLAY_PROTOCOL = {
    'memory_topk_ratio': 0.05,
    'memory_fusion_lambda': 1.0,
    'memory_route_margin_threshold': None,
    'memory_min_class_members': 1,
    'max_patches_per_class': 20000,
}
REPLAY_OUTPUT_FILES = (
    'coverage_replay_summary.json',
    'coverage_eval_scores.csv',
    'coverage_per_class_metrics.csv',
    'pseudo_class_members.csv',
    'pseudo_classes.csv',
    'tail_adaptive_neighborhood_edges.csv',
)


def sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        while True:
            block = file.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with path.open('r', encoding='utf-8') as file:
        return json.load(file)


def canonical_json_sha256(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def manifest_entry(run_id, role, path):
    path = Path(path)
    return {
        'run_id': run_id,
        'role': role,
        'path': str(path.resolve()),
        'size_bytes': path.stat().st_size,
        'sha256': sha256(path),
    }


def normalized_test_suffix(path):
    normalized = str(path).replace('\\', '/')
    if '/test/' not in normalized:
        raise ValueError('test path has no /test/ suffix: {}'.format(path))
    return normalized.split('/test/', 1)[1]


def selected_runs(v5_root, guard_root):
    rows = []
    for dataset, setting in SCENARIOS:
        for repeat in range(1, 6):
            run_id = '{}_{}_seed{:02d}'.format(dataset, setting, repeat)
            replacement = GUARD_REPLACEMENTS.get(run_id)
            run_dir = guard_root / replacement if replacement else v5_root / '{}_full'.format(run_id)
            source = 'partition_guard_v1' if replacement else 'dependency_ablation_v5'
            rows.append({
                'run_id': run_id,
                'dataset': dataset,
                'setting': setting,
                'repeat': repeat,
                'source': source,
                'full_run_dir': str(run_dir.resolve()),
                'guard_replacement': bool(replacement),
            })
    if len(rows) != 30 or sum(row['guard_replacement'] for row in rows) != 3:
        raise RuntimeError('fixed protocol must contain 30 runs and three Guard replacements')
    return rows


def data_path(row, mvtec_root, visa_root):
    if row['dataset'] == 'visa':
        return visa_root
    dataset_name = 'mvtecad-{}-seed{:02d}'.format(row['setting'], row['repeat'])
    return mvtec_root / dataset_name


def source_paths(run_dir):
    tailguard = run_dir / 'tailguard'
    return {
        'retained': tailguard / 'stage2/stage2_retained_samples.csv',
        'removed': tailguard / 'stage2/stage2_removed_samples.csv',
        'full_members': tailguard / 'pseudoclasses/pseudo_class_members.csv',
        'refined_scores': tailguard / 'memory/memory_eval_scores.csv',
        'refined_summary': tailguard / 'memory/memory_eval_summary.json',
        'tail_labels': tailguard / 'prepare/tail_sampler_analysis_only/sampler_analysis_details.csv',
        'tailguard_summary': tailguard / 'tailguard_summary.json',
        'attachment_config': tailguard / 'attachment_replay_config.json',
        'gbps_trigger': tailguard / 'gbps/gbps_trigger_summary.json',
    }


def normalized_method_config(summary):
    config = dict(summary['resolved_method_config'])
    config.setdefault('tailsampler_plateau_gap_guard', True)
    config.setdefault('tg_memory_route_margin_threshold', None)
    config.setdefault('tg_memory_min_class_members', 1)
    return config


def validate_tailguard_summary(row, summary):
    run_id = row['run_id']
    require(summary.get('method_mode') == 'full', '{} is not a Full run'.format(run_id))
    require(summary.get('config_profile') == 'final', '{} is not the final profile'.format(run_id))
    require(summary.get('memory_status') == 'completed', '{} memory evaluation is incomplete'.format(run_id))
    provenance = summary.get('dataset_provenance', {})
    require(provenance.get('profile_name') == row['dataset'], '{} dataset profile mismatch'.format(run_id))
    config = normalized_method_config(summary)
    stage2 = summary.get('stage2_summary', {})
    require(config.get('reconciliation_enabled') is True, '{} has reconciliation disabled'.format(run_id))
    require(config.get('enhancement_enabled') is True, '{} has enhancement disabled'.format(run_id))
    require(stage2.get('reconciliation_performed') is True, '{} did not perform reconciliation'.format(run_id))
    require(stage2.get('enhancement_performed') is True, '{} did not perform enhancement'.format(run_id))
    require(stage2.get('num_tail_candidates_removed') == 0, '{} removed protected candidates'.format(run_id))
    require(config.get('tg_memory_fusion_lambda') == 1.0, '{} has an unexpected lambda'.format(run_id))
    require(config.get('tg_memory_topk_ratio') == 0.05, '{} has an unexpected top-k ratio'.format(run_id))
    require(config.get('tg_memory_route_margin_threshold') is None, '{} has a route-margin gate'.format(run_id))
    require(config.get('tg_memory_min_class_members') == 1, '{} has an unexpected minimum class size'.format(run_id))
    require(config.get('tg_mem_max_patches_per_class') == 20000, '{} has an unexpected memory cap'.format(run_id))
    return config


def load_guard_audit(rows, guard_audit_csv):
    audit_path = Path(guard_audit_csv).resolve()
    if not audit_path.is_file():
        raise FileNotFoundError('TailSampler Guard audit is missing: {}'.format(audit_path))
    frame = pd.read_csv(audit_path)
    frame = frame.loc[frame['method'] == 'plateau_gap_guard'].copy()
    require(len(frame) == 30, 'Guard audit must contain 30 plateau-gap rows')
    require(frame['run'].is_unique, 'Guard audit contains duplicate runs')
    by_run = frame.set_index('run')
    expected_logical_runs = {'{}_full'.format(row['run_id']) for row in rows}
    require(set(by_run.index) == expected_logical_runs, 'Guard audit run inventory differs from fixed30')
    triggered = {
        str(name) for name, value in by_run['status'].items()
        if str(value).startswith('guarded;')
    }
    expected_triggered = {'{}_full'.format(run_id) for run_id in GUARD_REPLACEMENTS}
    require(triggered == expected_triggered, 'Guard trigger set differs from fixed replacements')
    records = []
    for row in rows:
        logical_run = '{}_full'.format(row['run_id'])
        audit = by_run.loc[logical_run]
        is_triggered = logical_run in triggered
        require(is_triggered == row['guard_replacement'], '{} Guard source selection mismatch'.format(row['run_id']))
        records.append({
            'run_id': row['run_id'],
            'logical_run': logical_run,
            'selected_source': row['source'],
            'guard_triggered': bool(is_triggered),
            'guard_status': str(audit['status']),
            'guard_cutoff': float(audit['cutoff']),
            'num_selected': int(audit['num_selected']),
            'selected_ratio': float(audit['selected_ratio']),
        })
    return records, manifest_entry('fixed30', 'guard_trigger_inventory', audit_path)


def validate_selected_runs(rows, guard_audit_csv):
    manifest = []
    semantic_config = None
    semantic_config_sha = None
    summaries = {}
    for row in rows:
        paths = source_paths(Path(row['full_run_dir']))
        for role, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError('{} missing for {}: {}'.format(role, row['run_id'], path))
            manifest.append(manifest_entry(row['run_id'], role, path))
        summary = read_json(paths['tailguard_summary'])
        config = validate_tailguard_summary(row, summary)
        config_sha = canonical_json_sha256(config)
        if semantic_config is None:
            semantic_config = config
            semantic_config_sha = config_sha
        require(config == semantic_config, '{} semantic method config differs'.format(row['run_id']))
        require(config_sha == semantic_config_sha, '{} semantic config hash differs'.format(row['run_id']))
        row['semantic_config_sha256'] = config_sha
        summaries[row['run_id']] = summary
    guard_records, guard_manifest = load_guard_audit(rows, guard_audit_csv)
    manifest.append(guard_manifest)
    for record in guard_records:
        row = next(item for item in rows if item['run_id'] == record['run_id'])
        summary = summaries[row['run_id']]
        source_config = summary['resolved_method_config']
        embedded = summary.get('prepare_summary', {}).get('tail_partition_guard')
        if row['guard_replacement']:
            require(source_config.get('tailsampler_plateau_gap_guard') is True, '{} Guard is not enabled'.format(row['run_id']))
            require(isinstance(embedded, dict) and embedded.get('triggered') is True, '{} Guard did not trigger'.format(row['run_id']))
            require(int(embedded.get('num_selected_final')) == record['num_selected'], '{} Guard selected-count mismatch'.format(row['run_id']))
            require(abs(float(embedded.get('effective_cutoff')) - record['guard_cutoff']) <= 1e-12, '{} Guard cutoff mismatch'.format(row['run_id']))
        else:
            require(not record['guard_triggered'], '{} unexpectedly requires a Guard replacement'.format(row['run_id']))
        record['source_guard_enabled'] = bool(source_config.get('tailsampler_plateau_gap_guard', False))
        record['embedded_guard_triggered'] = bool(embedded.get('triggered')) if isinstance(embedded, dict) else False
        record['semantic_config_sha256'] = semantic_config_sha
    return manifest, guard_records, semantic_config


def class_role_map(tail_labels, run_id):
    required = {'class_name', 'is_gt_tail'}
    missing = required.difference(tail_labels.columns)
    require(not missing, '{} tail labels miss {}'.format(run_id, sorted(missing)))
    role_counts = tail_labels.groupby('class_name')['is_gt_tail'].nunique(dropna=False)
    require(bool((role_counts == 1).all()), '{} has inconsistent class roles'.format(run_id))
    return {
        str(class_name): bool(int(values.iloc[0]))
        for class_name, values in tail_labels.groupby('class_name')['is_gt_tail']
    }


def validate_score_identity(coverage, refined, roles, run_id):
    key_columns = ['class_name', 'test_suffix', 'label']
    frames = []
    for name, frame in (('coverage', coverage), ('refined', refined)):
        keyed = frame.copy()
        require({'class_name', 'img_path', 'label', 'final_score'}.issubset(keyed.columns), '{} {} scores miss required columns'.format(run_id, name))
        keyed['class_name'] = keyed['class_name'].astype(str)
        keyed['test_suffix'] = keyed['img_path'].map(normalized_test_suffix)
        keyed['label'] = keyed['label'].astype(int)
        require(not keyed.duplicated(key_columns).any(), '{} {} scores contain duplicate test identities'.format(run_id, name))
        require(set(keyed['class_name']) == set(roles), '{} {} test classes differ from role labels'.format(run_id, name))
        for class_name, class_frame in keyed.groupby('class_name'):
            require(set(class_frame['label']) == {0, 1}, '{} {} class {} lacks both test labels'.format(run_id, name, class_name))
        frames.append(keyed)
    coverage_keys = frames[0][key_columns].sort_values(key_columns).reset_index(drop=True)
    refined_keys = frames[1][key_columns].sort_values(key_columns).reset_index(drop=True)
    require(coverage_keys.equals(refined_keys), '{} coverage/refined test identities differ'.format(run_id))
    return frames


def validate_replay_output(row, parsed_args, encoder_checkpoint, encoder_sha):
    output_dir = Path(parsed_args.output_dir).resolve() / 'runs' / row['run_id']
    for name in REPLAY_OUTPUT_FILES:
        if not (output_dir / name).is_file():
            raise FileNotFoundError('{} replay output is missing: {}'.format(row['run_id'], output_dir / name))
    summary = read_json(output_dir / 'coverage_replay_summary.json')
    expected_data_path = data_path(row, Path(parsed_args.mvtec_root), Path(parsed_args.visa_root)).resolve()
    expected_cache = Path(parsed_args.mvtec_cache if row['dataset'] == 'mvtec' else parsed_args.visa_cache).resolve()
    require(summary.get('evaluator') == 'dinomaly_tailguard_coverage_replay', '{} evaluator mismatch'.format(row['run_id']))
    require(summary.get('no_training') is True, '{} replay unexpectedly trained'.format(row['run_id']))
    require(Path(summary['full_run_dir']).resolve() == Path(row['full_run_dir']).resolve(), '{} Full source mismatch'.format(row['run_id']))
    require(summary.get('dataset_profile') == row['dataset'], '{} replay profile mismatch'.format(row['run_id']))
    require(Path(summary['data_path']).resolve() == expected_data_path, '{} replay data path mismatch'.format(row['run_id']))
    require(Path(summary['checkpoint_path']).resolve() == encoder_checkpoint.resolve(), '{} replay checkpoint path mismatch'.format(row['run_id']))
    require(summary.get('checkpoint_sha256') == encoder_sha, '{} replay checkpoint hash mismatch'.format(row['run_id']))
    require(int(summary.get('checkpoint_iteration')) == 10000, '{} replay checkpoint iteration mismatch'.format(row['run_id']))
    require(Path(summary['feature_cache_dir']).resolve() == expected_cache, '{} replay cache path mismatch'.format(row['run_id']))
    require(summary.get('protocol') == REPLAY_PROTOCOL, '{} replay protocol mismatch'.format(row['run_id']))
    integrity = summary.get('registry', {}).get('integrity', {})
    require(integrity and all(value is True for value in integrity.values()), '{} registry integrity failed'.format(row['run_id']))
    retained = pd.read_csv(source_paths(Path(row['full_run_dir']))['retained'])
    removed = pd.read_csv(source_paths(Path(row['full_run_dir']))['removed'])
    require(int(summary['registry']['num_retained_samples']) == len(retained), '{} retained count mismatch'.format(row['run_id']))
    require(int(summary['registry']['num_removed_samples']) == len(removed), '{} removed count mismatch'.format(row['run_id']))
    coverage = pd.read_csv(output_dir / 'coverage_eval_scores.csv')
    refined = pd.read_csv(source_paths(Path(row['full_run_dir']))['refined_scores'])
    labels = pd.read_csv(source_paths(Path(row['full_run_dir']))['tail_labels'])
    roles = class_role_map(labels, row['run_id'])
    coverage_keyed, _ = validate_score_identity(coverage, refined, roles, row['run_id'])
    require(int(summary.get('num_test_images')) == len(coverage), '{} test-image count mismatch'.format(row['run_id']))
    per_class = pd.read_csv(output_dir / 'coverage_per_class_metrics.csv')
    require(per_class['class_name'].astype(str).is_unique, '{} replay per-class rows are duplicated'.format(row['run_id']))
    require(set(per_class['class_name'].astype(str)) == set(roles), '{} replay per-class class set mismatch'.format(row['run_id']))
    recomputed = []
    for class_name, frame in coverage_keyed.groupby('class_name', sort=True):
        recomputed.append(float(roc_auc_score(frame['label'], frame['final_score'])))
    require(abs(float(np.mean(recomputed)) - float(summary['coverage_I-AUROC'])) <= 1e-12, '{} replay AUROC summary mismatch'.format(row['run_id']))
    return output_dir


def validate_encoder_metadata(encoder_checkpoint, expected_dataset):
    import torch

    payload = torch.load(str(encoder_checkpoint), map_location='cpu')
    require(int(payload.get('iteration')) == 10000, '{} shared encoder is not iteration 10000'.format(expected_dataset))
    checkpoint_args = payload.get('args') or {}
    if isinstance(checkpoint_args, dict):
        dataset_profile = checkpoint_args.get('dataset_profile')
        image_size = checkpoint_args.get('image_size', 448)
        crop_size = checkpoint_args.get('crop_size', 392)
    else:
        dataset_profile = getattr(checkpoint_args, 'dataset_profile', None)
        image_size = getattr(checkpoint_args, 'image_size', 448)
        crop_size = getattr(checkpoint_args, 'crop_size', 392)
    require(dataset_profile == expected_dataset, '{} shared encoder profile mismatch'.format(expected_dataset))
    require(int(image_size) == 448 and int(crop_size) == 392, '{} shared encoder preprocessing mismatch'.format(expected_dataset))
    del payload


def run_replay(row, parsed_args, encoder_checkpoint, encoder_sha):
    output_dir = Path(parsed_args.output_dir).resolve() / 'runs' / row['run_id']
    summary_path = output_dir / 'coverage_replay_summary.json'
    if summary_path.is_file() and parsed_args.skip_existing:
        validate_replay_output(row, parsed_args, encoder_checkpoint, encoder_sha)
        print('validated existing {}'.format(row['run_id']), flush=True)
        return output_dir
    if output_dir.exists():
        raise FileExistsError('partial/existing output requires manual audit: {}'.format(output_dir))
    dataset_path = data_path(row, Path(parsed_args.mvtec_root), Path(parsed_args.visa_root))
    cache_dir = Path(
        parsed_args.mvtec_cache if row['dataset'] == 'mvtec' else parsed_args.visa_cache
    ).resolve()
    command = [
        parsed_args.python,
        str((Path(__file__).resolve().parent / 'dinomaly_tailguard_coverage_replay.py')),
        '--full_run_dir', row['full_run_dir'],
        '--data_path', str(dataset_path.resolve()),
        '--encoder_checkpoint_path', str(encoder_checkpoint.resolve()),
        '--output_dir', str(output_dir),
        '--feature_cache_dir', str(cache_dir),
        '--dataset_profile', row['dataset'],
        '--gpu', str(parsed_args.gpu),
        '--batch_size', str(parsed_args.batch_size),
        '--no-save-memory-system',
    ]
    print('run {}'.format(row['run_id']), flush=True)
    subprocess.run(command, check=True, env=os.environ.copy())
    validate_replay_output(row, parsed_args, encoder_checkpoint, encoder_sha)
    return output_dir


def per_class_dual_metrics(row, coverage_scores, refined_scores, tail_labels):
    key_columns = ['class_name', 'test_suffix', 'label']
    roles = class_role_map(tail_labels, row['run_id'])
    coverage, refined = validate_score_identity(
        coverage_scores, refined_scores, roles, row['run_id']
    )
    merged = coverage[key_columns + ['final_score']].merge(
        refined[key_columns + ['final_score']],
        on=key_columns,
        suffixes=('_coverage', '_refined'),
        validate='one_to_one',
    )
    if len(merged) != len(coverage) or len(merged) != len(refined):
        raise ValueError('coverage/refined score identities do not align for {}'.format(row['run_id']))
    merged['dual_final_score'] = 0.5 * (
        merged['final_score_coverage'] + merged['final_score_refined']
    )
    class_rows = []
    for class_name, frame in merged.groupby('class_name', sort=True):
        class_rows.append({
            'run_id': row['run_id'],
            'dataset': row['dataset'],
            'setting': row['setting'],
            'repeat': row['repeat'],
            'class_name': str(class_name),
            'class_role': 'tail' if roles[str(class_name)] else 'head',
            'I-AUROC': float(roc_auc_score(frame['label'], frame['dual_final_score'])),
            'num_test_images': len(frame),
        })
    return pd.DataFrame(class_rows), merged


def aggregate_outputs(rows, output_root, source_manifest):
    per_class_frames = []
    score_manifest = []
    for row in rows:
        run_output = output_root / 'runs' / row['run_id']
        coverage_path = run_output / 'coverage_eval_scores.csv'
        refined_path = source_paths(Path(row['full_run_dir']))['refined_scores']
        tail_path = source_paths(Path(row['full_run_dir']))['tail_labels']
        coverage = pd.read_csv(coverage_path)
        refined = pd.read_csv(refined_path)
        tail_labels = pd.read_csv(tail_path)
        per_class, merged = per_class_dual_metrics(row, coverage, refined, tail_labels)
        per_class_frames.append(per_class)
        score_manifest.extend([
            manifest_entry(row['run_id'], 'coverage_scores', coverage_path),
            manifest_entry(row['run_id'], 'coverage_replay_summary', run_output / 'coverage_replay_summary.json'),
            manifest_entry(row['run_id'], 'coverage_per_class', run_output / 'coverage_per_class_metrics.csv'),
            {
                'run_id': row['run_id'], 'role': 'dual_score_identity_audit',
                'path': '<computed-in-memory>', 'size_bytes': len(merged),
                'sha256': hashlib.sha256(
                    merged.sort_values(['class_name', 'test_suffix', 'label']).to_csv(index=False).encode('utf-8')
                ).hexdigest(),
            },
        ])
    per_class = pd.concat(per_class_frames, ignore_index=True)
    per_run_rows = []
    for keys, frame in per_class.groupby(['run_id', 'dataset', 'setting', 'repeat'], sort=True):
        run_id, dataset, setting, repeat = keys
        values = {}
        for role in ('tail', 'head'):
            subset = frame.loc[frame['class_role'] == role, 'I-AUROC']
            if subset.empty:
                raise ValueError('{} has no {} classes'.format(run_id, role))
            values['{}_I-AUROC'.format(role)] = float(subset.mean())
            values['{}_num_classes'.format(role)] = int(len(subset))
        values['all_I-AUROC'] = float(frame['I-AUROC'].mean())
        values['all_num_classes'] = int(len(frame))
        per_run_rows.append({
            'run_id': run_id, 'dataset': dataset, 'setting': setting,
            'repeat': int(repeat), **values,
        })
    per_run = pd.DataFrame(per_run_rows).sort_values(['dataset', 'setting', 'repeat'])
    scenario_rows = []
    metrics = ('tail_I-AUROC', 'head_I-AUROC', 'all_I-AUROC')
    for (dataset, setting), frame in per_run.groupby(['dataset', 'setting'], sort=True):
        if len(frame) != 5:
            raise ValueError('{} {} does not contain five repeats'.format(dataset, setting))
        payload = {'dataset': dataset, 'setting': setting, 'num_repeats': len(frame)}
        for metric in metrics:
            payload['{}_mean'.format(metric)] = float(frame[metric].mean())
            payload['{}_sample_sd'.format(metric)] = float(frame[metric].std(ddof=1))
        scenario_rows.append(payload)
    scenario = pd.DataFrame(scenario_rows)
    aggregate_rows = []
    for scope, frame in [('all_scenarios', per_run)]:
        payload = {'scope': scope, 'num_scenarios': len(scenario), 'num_runs': len(frame)}
        for metric in metrics:
            scenario_means = scenario['{}_mean'.format(metric)]
            repeat_macro = frame.groupby('repeat')[metric].mean()
            payload['{}_scenario_macro_mean'.format(metric)] = float(scenario_means.mean())
            payload['{}_repeat_macro_sample_sd'.format(metric)] = float(repeat_macro.std(ddof=1))
        aggregate_rows.append(payload)
    aggregate = pd.DataFrame(aggregate_rows)

    per_class.to_csv(output_root / 'dual_per_class.csv', index=False)
    per_run.to_csv(output_root / 'dual_per_run.csv', index=False)
    scenario.to_csv(output_root / 'dual_by_scenario.csv', index=False)
    aggregate.to_csv(output_root / 'dual_aggregate.csv', index=False)
    pd.DataFrame(source_manifest + score_manifest).to_csv(output_root / 'source_manifest.csv', index=False)
    return per_run, scenario, aggregate


def validate_visa_mapping_audit(rows, output_root, visa_root):
    paths = {
        'visa_mapping_summary': output_root / 'visa_source_mapping_summary.json',
        'visa_mapping_rows': output_root / 'visa_source_mapping.csv',
        'visa_mapping_by_run': output_root / 'visa_source_mapping_by_run.csv',
        'visa_mapping_by_setting': output_root / 'visa_source_mapping_by_setting.csv',
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError('{} is missing: {}'.format(role, path))
    summary = read_json(paths['visa_mapping_summary'])
    require(summary.get('status') == 'passed', 'VisA source mapping audit did not pass')
    require(int(summary.get('num_runs')) == 15, 'VisA source mapping audit run count mismatch')
    require(int(summary.get('num_mapping_errors')) == 0, 'VisA source mapping audit has errors')
    require(Path(summary.get('source_root')).resolve() == Path(visa_root).resolve(), 'VisA source root mismatch')
    by_run = pd.read_csv(paths['visa_mapping_by_run'])
    expected_runs = {row['run_id'] for row in rows if row['dataset'] == 'visa'}
    require(set(by_run['run_id']) == expected_runs, 'VisA source mapping run inventory mismatch')
    require(int(by_run['num_retained'].sum()) == int(summary['num_mapped_samples']), 'VisA source mapping sample count mismatch')
    return [manifest_entry('fixed30', role, path) for role, path in paths.items()]


def validate_pixel_statistics(rows, pixel_summary_dir):
    root = Path(pixel_summary_dir).resolve()
    paths = {
        'pixel_selected_runs': root / 'selected_runs.csv',
        'pixel_scenario_summary': root / 'scenario_summary.csv',
        'pixel_aggregate_summary': root / 'aggregate_summary.json',
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError('{} is missing: {}'.format(role, path))
    selected = pd.read_csv(paths['pixel_selected_runs'])
    require(len(selected) == 30 and selected['logical_run'].is_unique, 'pixel statistics do not select 30 unique runs')
    selected = selected.assign(
        run_id=selected['logical_run'].str.replace(r'_full$', '', regex=True)
    ).set_index('run_id')
    require(set(selected.index) == {row['run_id'] for row in rows}, 'pixel/image run inventories differ')
    for row in rows:
        pixel_row = selected.loc[row['run_id']]
        require(Path(pixel_row['run_dir']).resolve() == Path(row['full_run_dir']).resolve(), '{} pixel/image sources differ'.format(row['run_id']))
        require(bool(pixel_row['uses_partition_guard']) == row['guard_replacement'], '{} pixel/image Guard selection differs'.format(row['run_id']))
    scenario = pd.read_csv(paths['pixel_scenario_summary'])
    require(len(scenario) == 18, 'pixel scenario summary must contain six settings by three roles')
    require(set(scenario['role']) == {'tail', 'head', 'all'}, 'pixel scenario roles are incomplete')
    aggregate = read_json(paths['pixel_aggregate_summary'])
    require(int(aggregate.get('num_runs')) == 30, 'pixel aggregate run count mismatch')
    require(int(aggregate.get('num_scenarios')) == 6, 'pixel aggregate scenario count mismatch')
    require(int(aggregate.get('num_constructions_per_scenario')) == 5, 'pixel aggregate repeat count mismatch')
    require(aggregate.get('selection_used_test_labels') is False, 'pixel run selection used test labels')
    return scenario, aggregate, [
        manifest_entry('fixed30', role, path) for role, path in paths.items()
    ]


def write_paper_ready_statistics(output_root, image_scenario, image_aggregate, pixel_scenario, pixel_aggregate):
    rows = []
    for _, image_row in image_scenario.iterrows():
        for role in ('tail', 'head', 'all'):
            pixel = pixel_scenario.loc[
                (pixel_scenario['dataset'] == image_row['dataset'])
                & (pixel_scenario['setting'] == image_row['setting'])
                & (pixel_scenario['role'] == role)
            ]
            require(len(pixel) == 1, '{} {} {} pixel summary is not unique'.format(image_row['dataset'], image_row['setting'], role))
            pixel = pixel.iloc[0]
            rows.append({
                'dataset': image_row['dataset'],
                'setting': image_row['setting'],
                'role': role,
                'num_repeats': 5,
                'I-AUROC_mean_percent': 100.0 * float(image_row['{}_I-AUROC_mean'.format(role)]),
                'I-AUROC_sample_sd_percent': 100.0 * float(image_row['{}_I-AUROC_sample_sd'.format(role)]),
                'P-AUROC_mean_percent': 100.0 * float(pixel['P-AUROC_mean']),
                'P-AUROC_sample_sd_percent': 100.0 * float(pixel['P-AUROC_std']),
                'P-AUPRO_mean_percent': 100.0 * float(pixel['P-AUPRO_mean']),
                'P-AUPRO_sample_sd_percent': 100.0 * float(pixel['P-AUPRO_std']),
            })
    paper_scenario = pd.DataFrame(rows)
    paper_scenario.to_csv(output_root / 'paper_ready_by_scenario.csv', index=False)

    image = image_aggregate.iloc[0]
    overall_rows = []
    for role in ('tail', 'head', 'all'):
        pixel = pixel_aggregate['five_construction_summary'][role]
        overall_rows.append({
            'scope': 'six_scenario_macro',
            'role': role,
            'num_repeats': 5,
            'I-AUROC_mean_percent': 100.0 * float(image['{}_I-AUROC_scenario_macro_mean'.format(role)]),
            'I-AUROC_sample_sd_percent': 100.0 * float(image['{}_I-AUROC_repeat_macro_sample_sd'.format(role)]),
            'P-AUROC_mean_percent': 100.0 * float(pixel['P-AUROC']['mean']),
            'P-AUROC_sample_sd_percent': 100.0 * float(pixel['P-AUROC']['std']),
            'P-AUPRO_mean_percent': 100.0 * float(pixel['P-AUPRO']['mean']),
            'P-AUPRO_sample_sd_percent': 100.0 * float(pixel['P-AUPRO']['std']),
        })
    paper_overall = pd.DataFrame(overall_rows)
    paper_overall.to_csv(output_root / 'paper_ready_aggregate.csv', index=False)
    return paper_scenario, paper_overall


def main(parsed_args):
    output_root = Path(parsed_args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = selected_runs(Path(parsed_args.v5_root), Path(parsed_args.guard_root))
    source_manifest, guard_records, semantic_config = validate_selected_runs(
        rows, parsed_args.guard_audit_csv
    )
    encoder_checkpoints = {
        dataset: Path(parsed_args.v5_root) / '{}_pareto_seed01_full/final_model.pt'.format(dataset)
        for dataset in ('mvtec', 'visa')
    }
    encoder_hashes = {}
    for dataset, checkpoint in encoder_checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError('{} encoder checkpoint missing: {}'.format(dataset, checkpoint))
        validate_encoder_metadata(checkpoint, dataset)
        encoder_hashes[dataset] = sha256(checkpoint)
        source_manifest.append(manifest_entry(
            '{}_shared_encoder'.format(dataset), 'encoder_checkpoint', checkpoint
        ))
    source_manifest.extend(validate_visa_mapping_audit(
        rows, output_root, parsed_args.visa_root
    ))
    pixel_scenario, pixel_aggregate, pixel_manifest = validate_pixel_statistics(
        rows, parsed_args.pixel_summary_dir
    )
    source_manifest.extend(pixel_manifest)
    selected = pd.DataFrame(rows)
    selected.to_csv(output_root / 'selected_runs.csv', index=False)
    pd.DataFrame(guard_records).to_csv(output_root / 'guard_selection_audit.csv', index=False)
    write_json(output_root / 'semantic_method_config.json', {
        'normalized_config': semantic_config,
        'sha256': canonical_json_sha256(semantic_config),
    })
    write_json(output_root / 'fixed_protocol.json', {
        'schema_version': 2,
        'num_runs': len(rows),
        'guard_replacements': GUARD_REPLACEMENTS,
        'dual_rule': '0.5 * (coverage_final_score + refined_final_score)',
        'class_metric': 'per-class image AUROC',
        'role_aggregation': 'macro mean over ground-truth tail/head classes',
        'repeat_aggregation': 'mean and sample standard deviation (ddof=1)',
        'pixel_protocol': 'reconciled final anomaly map from the same fixed 30 selected runs',
        'replay_protocol': REPLAY_PROTOCOL,
        'semantic_method_config_sha256': canonical_json_sha256(semantic_config),
        'encoder_sha256': encoder_hashes,
    })
    for row in rows:
        run_replay(
            row,
            parsed_args,
            encoder_checkpoints[row['dataset']],
            encoder_hashes[row['dataset']],
        )
    source_manifest.extend([
        manifest_entry('fixed30', 'selected_run_inventory', output_root / 'selected_runs.csv'),
        manifest_entry('fixed30', 'guard_selection_audit', output_root / 'guard_selection_audit.csv'),
        manifest_entry('fixed30', 'semantic_method_config', output_root / 'semantic_method_config.json'),
        manifest_entry('fixed30', 'fixed_protocol', output_root / 'fixed_protocol.json'),
    ])
    per_run, scenario, aggregate = aggregate_outputs(rows, output_root, source_manifest)
    write_paper_ready_statistics(
        output_root, scenario, aggregate, pixel_scenario, pixel_aggregate
    )
    print(scenario.to_string(index=False), flush=True)
    print(aggregate.to_string(index=False), flush=True)
    return per_run, scenario, aggregate


def build_parser():
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Fixed-30 coverage replay and dual aggregation')
    parser.add_argument('--v5_root', default='/home/linux/projects/results/tailguard/dependency_ablation_v5')
    parser.add_argument('--guard_root', default='/home/linux/projects/results/tailguard/partition_guard_v1')
    parser.add_argument('--mvtec_root', default='/home/linux/projects/LTN_datasets')
    parser.add_argument('--visa_root', default='/home/linux/projects/visa_')
    parser.add_argument('--output_dir', default=str(repo / 'analysis_outputs/coverage_replay_v1/fixed30'))
    parser.add_argument(
        '--mvtec_cache',
        default=str(repo / 'analysis_outputs/coverage_replay_v1/feature_cache_mvtec'),
    )
    parser.add_argument(
        '--visa_cache',
        default=str(repo / 'analysis_outputs/coverage_replay_v1/feature_cache_visa'),
    )
    parser.add_argument('--python', default='/home/linux/miniconda3/envs/dinomaly/bin/python')
    parser.add_argument(
        '--guard_audit_csv',
        default=str(repo / 'analysis_outputs/tailsampler_partition_guard_v1/per_run.csv'),
    )
    parser.add_argument(
        '--pixel_summary_dir',
        default=str(repo / 'analysis_outputs/tailguard_refined_only_five_constructions_v1'),
    )
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--skip_existing', action='store_true')
    return parser


if __name__ == '__main__':
    main(build_parser().parse_args())
