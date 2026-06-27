import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if torch.is_tensor(value):
        return make_json_safe(value.detach().cpu().tolist())
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_summary_json(path: str, summary: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as file:
        json.dump(make_json_safe(summary), file, indent=2)


def save_result_csv(path: str, result_df: pd.DataFrame) -> None:
    ensure_dir(os.path.dirname(path))
    result_df.to_csv(path, index=False)


def save_features_npz(path: str, arrays: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **{key: value for key, value in arrays.items() if value is not None})


def save_graph_npz(path: str, rows, cols, edge_scores, extras: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        'rows': np.asarray(rows, dtype=np.int64),
        'cols': np.asarray(cols, dtype=np.int64),
        'edge_scores': np.asarray(edge_scores, dtype=np.float32),
    }
    if extras is not None:
        for key, value in extras.items():
            payload[key] = np.asarray(value)
    save_features_npz(path, payload)
