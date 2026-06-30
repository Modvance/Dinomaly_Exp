from .config import PreclusterConfig, load_config
from .discovery import discover_semantic_prototypes


def run_precluster_visual(*args, **kwargs):
    from .run import run_precluster_visual as _run_precluster_visual
    return _run_precluster_visual(*args, **kwargs)


__all__ = ['PreclusterConfig', 'load_config', 'discover_semantic_prototypes', 'run_precluster_visual']
