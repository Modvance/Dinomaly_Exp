import argparse
import os

from precluster_visual.config import load_config
from precluster_visual.run import run_precluster_visual


def parse_args():
    parser = argparse.ArgumentParser(description='Run standalone semantic prototype discovery without labels or fixed class count')
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--stage', type=str, default='all', choices=['extract', 'cluster', 'all'])
    parser.add_argument('--split', type=str, default=None, choices=['train', 'test', 'all'])
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--input_size', type=int, default=None)
    parser.add_argument('--preprocess_mode', type=str, default=None, choices=['default', 'letterbox'])
    parser.add_argument('--margin_threshold', type=float, default=None)
    parser.add_argument('--max_supports', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.split is not None:
        overrides.setdefault('data', {})['split'] = args.split
    if args.model_name is not None:
        overrides.setdefault('encoder', {})['model_name'] = args.model_name
    if args.device is not None:
        overrides.setdefault('encoder', {})['device'] = args.device
    if args.batch_size is not None:
        overrides.setdefault('encoder', {})['batch_size'] = args.batch_size
    if args.num_workers is not None:
        overrides.setdefault('encoder', {})['num_workers'] = args.num_workers
    if args.input_size is not None:
        overrides.setdefault('encoder', {})['input_size'] = args.input_size
    if args.preprocess_mode is not None:
        overrides.setdefault('preprocess', {})['mode'] = args.preprocess_mode
    if args.margin_threshold is not None:
        overrides.setdefault('assignment', {})['margin_threshold'] = args.margin_threshold
    if args.max_supports is not None:
        overrides.setdefault('prototype', {})['max_supports'] = args.max_supports

    config = load_config(
        config_path=args.config,
        overrides=overrides,
        image_root=os.path.abspath(args.image_root),
        output_dir=os.path.abspath(args.output_dir),
    )
    result = run_precluster_visual(config, stage=args.stage)
    print('stage={}'.format(args.stage))
    print('num_samples={}'.format(result['num_samples']))
    if 'num_clusters' in result:
        print('num_clusters={}'.format(result['num_clusters']))
    print('output_dir={}'.format(result['output_dir']))
    if 'features_path' in result:
        print('features_path={}'.format(result['features_path']))
    if 'result_csv_path' in result:
        print('result_csv_path={}'.format(result['result_csv_path']))
    if 'summary_json_path' in result:
        print('summary_json_path={}'.format(result['summary_json_path']))
    if 'prototype_npz_path' in result:
        print('prototype_npz_path={}'.format(result['prototype_npz_path']))


if __name__ == '__main__':
    main()
