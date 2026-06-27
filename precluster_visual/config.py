from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json


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
    model_name: str = 'dinov2_vitb14'
    pretrained: bool = True
    input_size: int = 518
    batch_size: int = 16
    num_workers: int = 4
    device: str = 'cuda'
    precision: str = 'fp16'
    use_registers: bool = False
    high_layer: int = -1
    mid_layers: List[int] = field(default_factory=lambda: [-6, -5, -4])


@dataclass
class FeatureConfig:
    structure_grid: List[int] = field(default_factory=lambda: [2, 2])
    object_weight: float = 1.0
    structure_weight: float = 0.7
    texture_weight: float = 0.5
    normalize_each_feature: bool = True
    normalize_final_feature: bool = True
    foreground_eps: float = 1e-6


@dataclass
class MultiViewConfig:
    enabled: bool = True
    num_views: int = 3
    scale_min: float = 0.9
    scale_max: float = 1.0
    brightness: float = 0.1
    contrast: float = 0.1


@dataclass
class KnnConfig:
    metric: str = 'cosine'
    k_min: int = 40
    k_max: int = 140
    mutual: bool = True
    use_snn: bool = True


@dataclass
class GraphConfig:
    max_edges_per_node: int = 10


@dataclass
class ClusterConfig:
    small_cluster_size: int = 3
    keep_singleton: bool = True


@dataclass
class DebugConfig:
    save_feature_parts: bool = True
    save_graph: bool = True
    random_seed: int = 42


@dataclass
class PreclusterConfig:
    data: DataConfig
    output: OutputConfig
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    multiview: MultiViewConfig = field(default_factory=MultiViewConfig)
    knn: KnnConfig = field(default_factory=KnnConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    clustering: ClusterConfig = field(default_factory=ClusterConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path) as file:
        return json.load(file)


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
        features=FeatureConfig(**data.get('features', {})),
        multiview=MultiViewConfig(**data.get('multiview', {})),
        knn=KnnConfig(**data.get('knn', {})),
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
        loaded = _read_json(Path(config_path))
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
    if config.knn.k_min <= 0 or config.knn.k_max <= 0:
        raise ValueError('k_min and k_max must be positive')
    if config.knn.k_min > config.knn.k_max:
        raise ValueError('k_min cannot exceed k_max')
    if len(config.features.structure_grid) != 2:
        raise ValueError('structure_grid must have two integers')
    if any(int(v) <= 0 for v in config.features.structure_grid):
        raise ValueError('structure_grid values must be positive')
    if config.graph.max_edges_per_node <= 0:
        raise ValueError('graph.max_edges_per_node must be positive')
    if config.multiview.enabled and config.multiview.num_views <= 0:
        raise ValueError('multiview.num_views must be positive when multiview is enabled')
