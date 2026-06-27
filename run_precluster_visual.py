import argparse
import os

from precluster_visual.config import load_config
from precluster_visual.run import run_precluster_visual


def parse_args():
    parser = argparse.ArgumentParser(description='Run standalone visual preclustering without labels or fixed class count')
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--split', type=str, default=None, choices=['train', 'test', 'all'])
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--input_size', type=int, default=None)
    parser.add_argument('--max_edges_per_node', type=int, default=None)
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
    if args.max_edges_per_node is not None:
        overrides.setdefault('graph', {})['max_edges_per_node'] = args.max_edges_per_node

    config = load_config(
        config_path=args.config,
        overrides=overrides,
        image_root=os.path.abspath(args.image_root),
        output_dir=os.path.abspath(args.output_dir),
    )
    result = run_precluster_visual(config)
    print('num_samples={}'.format(result['num_samples']))
    print('num_clusters={}'.format(result['num_clusters']))
    print('output_dir={}'.format(result['output_dir']))


if __name__ == '__main__':
    main()
