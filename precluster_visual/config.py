from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

try:
    import yaml
except ImportError:
    yaml = None


_DEFAULT_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']


@dataclass
class DataConfig:
    image_root: str
    image_extensions: List[str] = field(default_factory=lambda: list(_DEFAULT_EXTENSIONS))
    recursive: bool = True
    split: str = 'train'


@dataclass
class OutputConfig:
    output_dir: str
    result_csv: str = 'precluster_visual_result.csv'
    summary_json: str = 'precluster_visual_summary.json'
    features_npz: str = 'precluster_visual_features.npz'
    graph_npz: str = 'precluster_visual_graph.npz'


@dataclass
class EncoderConfig:
    model_name: str = 'google/siglip-large-patch16-384'
    input_size: int = 384
    batch_size: int = 16
    num_workers: int = 4
    device: str = 'cuda'
    precision: str = 'fp16'
    normalize: bool = True


@dataclass
class MultiViewConfig:
    enabled: bool = True
    num_views: int = 3
    scale_min: float = 0.92
    scale_max: float = 1.0
    brightness: float = 0.1
    contrast: float = 0.1
    saturation: float = 0.05
    hue: float = 0.02


@dataclass
class KnnConfig:
    metric: str = 'cosine'
    k_rule: str = 'sqrt'
    k_min: int = 20
    k_max: int = 140
    mutual: bool = True
    use_snn: bool = False


@dataclass
class CalibrationConfig:
    enabled: bool = True
    num_bins: int = 100
    num_negative_pairs: int = 50000
    connect_if_p_same_gt: float = 0.45


@dataclass
class GraphConfig:
    max_edges_per_node: int = 12
    use_local_top_edges: bool = True


@dataclass
class ClusterConfig:
    small_cluster_size: int = 3
    keep_singleton: bool = True


@dataclass
class DebugConfig:
    save_pair_statistics: bool = True
    save_graph: bool = True
    random_seed: int = 42


@dataclass
class PreclusterConfig:
    data: DataConfig
    output: OutputConfig
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    multiview: MultiViewConfig = field(default_factory=MultiViewConfig)
    knn: KnnConfig = field(default_factory=KnnConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    clustering: ClusterConfig = field(default_factory=ClusterConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_config(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    with open(path) as file:
        if suffix == '.json':
            return json.load(file)
        if suffix in ['.yaml', '.yml']:
            if yaml is None:
                raise ImportError('PyYAML is required to load YAML config files')
            return yaml.safe_load(file)
    raise ValueError('unsupported config extension: {}'.format(path.suffix))


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _coerce_dataclass(data: Dict[str, Any]) -> PreclusterConfig:
    return PreclusterConfig(
        data=DataConfig(**data['data']),
        output=OutputConfig(**data['output']),
        encoder=EncoderConfig(**data.get('encoder', {})),
        multiview=MultiViewConfig(**data.get('multiview', {})),
        knn=KnnConfig(**data.get('knn', {})),
        calibration=CalibrationConfig(**data.get('calibration', {})),
        graph=GraphConfig(**data.get('graph', {})),
        clustering=ClusterConfig(**data.get('clustering', {})),
        debug=DebugConfig(**data.get('debug', {})),
    )


def load_config(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None,
                image_root: Optional[str] = None, output_dir: Optional[str] = None) -> PreclusterConfig:
    base = PreclusterConfig(
        data=DataConfig(image_root=image_root or ''),
        output=OutputConfig(output_dir=output_dir or ''),
    ).to_dict()
    if config_path is not None:
        loaded = _read_config(Path(config_path))
        base = _merge_dict(base, loaded)
    if overrides is not None:
        base = _merge_dict(base, overrides)
    if image_root is not None:
        base['data']['image_root'] = image_root
    if output_dir is not None:
        base['output']['output_dir'] = output_dir
    config = _coerce_dataclass(base)
    validate_config(config)
    return config


def validate_config(config: PreclusterConfig) -> None:
    if config.data.image_root == '':
        raise ValueError('data.image_root is required')
    if config.output.output_dir == '':
        raise ValueError('output.output_dir is required')
    if str(config.data.split).lower() not in ['train', 'test', 'all']:
        raise ValueError('data.split must be one of: train, test, all')
    if config.encoder.batch_size <= 0:
        raise ValueError('encoder.batch_size must be positive')
    if config.encoder.num_workers < 0:
        raise ValueError('encoder.num_workers must be non-negative')
    if str(config.encoder.precision).lower() not in ['fp32', 'fp16', 'bf16']:
        raise ValueError('encoder.precision must be one of: fp32, fp16, bf16')
    if config.knn.k_rule not in ['sqrt']:
        raise ValueError('knn.k_rule must currently be sqrt')
    if config.knn.k_min <= 0 or config.knn.k_max <= 0:
        raise ValueError('k_min and k_max must be positive')
    if config.knn.k_min > config.knn.k_max:
        raise ValueError('k_min cannot exceed k_max')
    if config.calibration.num_bins <= 1:
        raise ValueError('calibration.num_bins must be greater than 1')
    if config.calibration.num_negative_pairs <= 0:
        raise ValueError('calibration.num_negative_pairs must be positive')
    if not 0.0 <= float(config.calibration.connect_if_p_same_gt) <= 1.0:
        raise ValueError('calibration.connect_if_p_same_gt must be within [0, 1]')
    if config.graph.max_edges_per_node <= 0:
        raise ValueError('graph.max_edges_per_node must be positive')
    if config.multiview.enabled and config.multiview.num_views <= 0:
        raise ValueError('multiview.num_views must be positive when multiview is enabled')
