"""Build a train-only TailGuard tri-state registry from saved v5 artifacts."""

import argparse
import hashlib
import json
import os

import pandas as pd

from tailguard_tri_state_registry import TriStatePolicy, build_tri_state_registry


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        while True:
            block = file.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _artifact_paths(run_dir):
    root = os.path.realpath(run_dir)
    return {
        'members': os.path.join(root, 'tailguard', 'pseudoclasses', 'pseudo_class_members.csv'),
        'classes': os.path.join(root, 'tailguard', 'pseudoclasses', 'pseudo_classes.csv'),
        'loo_report': os.path.join(root, 'tailguard', 'stage1_cleanup_pseudoclass_predictions.csv'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full_run_dir', required=True)
    parser.add_argument('--raw_e_run_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--disable_auxiliary_memory', action='store_true')
    parser.add_argument('--low_margin_threshold', type=float)
    parser.add_argument('--threshold_protocol_id')
    parser.add_argument(
        '--threshold_protocol_kind',
        choices=['independent_development', 'pre_registered', 'exploratory_target_audited'],
    )
    args = parser.parse_args()

    output_dir = os.path.realpath(args.output_dir)
    if os.path.exists(output_dir):
        raise FileExistsError('output_dir must not already exist: {}'.format(output_dir))
    full_paths = _artifact_paths(args.full_run_dir)
    raw_paths = _artifact_paths(args.raw_e_run_dir)
    required = [
        full_paths['members'], full_paths['classes'], full_paths['loo_report'],
        raw_paths['members'], raw_paths['classes'],
    ]
    for path in required:
        if not os.path.isfile(path):
            raise FileNotFoundError('required artifact does not exist: {}'.format(path))

    policy = TriStatePolicy(
        enable_auxiliary_memory=not args.disable_auxiliary_memory,
        low_margin_threshold=args.low_margin_threshold,
        threshold_protocol_id=args.threshold_protocol_id,
        threshold_protocol_kind=args.threshold_protocol_kind,
    )
    result = build_tri_state_registry(
        pd.read_csv(full_paths['members']),
        pd.read_csv(raw_paths['members']),
        pd.read_csv(full_paths['loo_report']),
        policy,
        full_classes=pd.read_csv(full_paths['classes']),
        raw_classes=pd.read_csv(raw_paths['classes']),
    )

    os.makedirs(output_dir)
    outputs = {
        'sample_roles': os.path.join(output_dir, 'tri_state_sample_roles.csv'),
        'routing_classes': os.path.join(output_dir, 'tri_state_routing_classes.csv'),
        'routing_members': os.path.join(output_dir, 'tri_state_routing_members.csv'),
        'memory_members': os.path.join(output_dir, 'tri_state_memory_members.csv'),
        'summary': os.path.join(output_dir, 'tri_state_registry_summary.json'),
    }
    result['sample_roles_df'].to_csv(outputs['sample_roles'], index=False)
    result['routing_classes_df'].to_csv(outputs['routing_classes'], index=False)
    result['routing_members_df'].to_csv(outputs['routing_members'], index=False)
    result['memory_members_df'].to_csv(outputs['memory_members'], index=False)
    summary = dict(result['summary'])
    summary['provenance'] = {
        'full_run_dir': os.path.realpath(args.full_run_dir),
        'raw_e_run_dir': os.path.realpath(args.raw_e_run_dir),
        'inputs': {
            path: _sha256(path) for path in required
        },
    }
    summary['outputs'] = outputs
    with open(outputs['summary'], 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()

