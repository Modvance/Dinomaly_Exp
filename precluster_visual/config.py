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
    prototype_npz: str = 'precluster_visual_prototypes.npz'


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
class PreprocessConfig:
    mode: str = 'default'
    letterbox_fill: int = 0


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
class CalibrationConfig:
    enabled: bool = True
    num_bins: int = 100
    num_negative_pairs: int = 50000
    base_threshold_delta: float = 0.05
    assign_threshold_floor_offset: float = 0.05


@dataclass
class PrototypeConfig:
    max_supports: int = 5
    support_top_r: int = 2
    min_members_for_adaptive_threshold: int = 3
    member_score_quantile: float = 0.10
    update_center_only_high_confidence: bool = False
    high_confidence_margin_bonus: float = 0.02


@dataclass
class AssignmentConfig:
    margin_threshold: float = 0.02
    use_class_adaptive_threshold: bool = True


@dataclass
class MergeConfig:
    enabled: bool = True
    merge_threshold_offset: float = 0.02
    support_top_r: int = 2
    max_merge_passes: int = 1


@dataclass
class ClusterConfig:
    small_cluster_size: int = 3
    keep_singleton: bool = True


@dataclass
class DebugConfig:
    save_pair_statistics: bool = True
    save_prototypes: bool = True
    random_seed: int = 42


@dataclass
class PreclusterConfig:
    data: DataConfig
    output: OutputConfig
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    multiview: MultiViewConfig = field(default_factory=MultiViewConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    prototype: PrototypeConfig = field(default_factory=PrototypeConfig)
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
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
        preprocess=PreprocessConfig(**data.get('preprocess', {})),
        multiview=MultiViewConfig(**data.get('multiview', {})),
        calibration=CalibrationConfig(**data.get('calibration', {})),
        prototype=PrototypeConfig(**data.get('prototype', {})),
        assignment=AssignmentConfig(**data.get('assignment', {})),
        merge=MergeConfig(**data.get('merge', {})),
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
    if int(config.encoder.input_size) <= 0:
        raise ValueError('encoder.input_size must be positive')
    if str(config.preprocess.mode).lower() not in ['default', 'letterbox']:
        raise ValueError('preprocess.mode must be one of: default, letterbox')
    if config.multiview.enabled and config.multiview.num_views <= 0:
        raise ValueError('multiview.num_views must be positive when multiview is enabled')
    if config.calibration.num_bins <= 1:
        raise ValueError('calibration.num_bins must be greater than 1')
    if config.calibration.num_negative_pairs <= 0:
        raise ValueError('calibration.num_negative_pairs must be positive')
    if not 0.0 <= float(config.calibration.base_threshold_delta) <= 1.0:
        raise ValueError('calibration.base_threshold_delta must be within [0, 1]')
    if float(config.calibration.assign_threshold_floor_offset) < 0.0:
        raise ValueError('calibration.assign_threshold_floor_offset must be non-negative')
    if int(config.prototype.max_supports) <= 0:
        raise ValueError('prototype.max_supports must be positive')
    if int(config.prototype.support_top_r) <= 0:
        raise ValueError('prototype.support_top_r must be positive')
    if int(config.prototype.min_members_for_adaptive_threshold) <= 0:
        raise ValueError('prototype.min_members_for_adaptive_threshold must be positive')
    if not 0.0 <= float(config.prototype.member_score_quantile) <= 1.0:
        raise ValueError('prototype.member_score_quantile must be within [0, 1]')
    if float(config.prototype.high_confidence_margin_bonus) < 0.0:
        raise ValueError('prototype.high_confidence_margin_bonus must be non-negative')
    if float(config.assignment.margin_threshold) < 0.0:
        raise ValueError('assignment.margin_threshold must be non-negative')
    if float(config.merge.merge_threshold_offset) < 0.0:
        raise ValueError('merge.merge_threshold_offset must be non-negative')
    if int(config.merge.support_top_r) <= 0:
        raise ValueError('merge.support_top_r must be positive')
    if int(config.merge.max_merge_passes) <= 0:
        raise ValueError('merge.max_merge_passes must be positive')
    if int(config.clustering.small_cluster_size) <= 0:
        raise ValueError('clustering.small_cluster_size must be positive')
