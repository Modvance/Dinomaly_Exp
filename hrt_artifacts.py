import json
import os
from typing import Dict, Optional

import pandas as pd
import torch


def save_hrt_artifacts(
    output_dir: str,
    summary: Dict,
    candidate_groups_df: pd.DataFrame,
    head_references_df: pd.DataFrame,
    sample_scores_df: pd.DataFrame,
    group_decisions_df: pd.DataFrame,
    final_selection_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    metadata: Dict,
    deviation_maps: Optional[Dict[int, torch.Tensor]] = None,
    patch_bank: Optional[Dict[int, Dict]] = None,
    save_patch_maps: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    candidate_groups_path = os.path.join(output_dir, 'hrt_candidate_groups.csv')
    head_references_path = os.path.join(output_dir, 'hrt_head_references.csv')
    sample_scores_path = os.path.join(output_dir, 'hrt_sample_scores.csv')
    group_decisions_path = os.path.join(output_dir, 'hrt_group_decisions.csv')
    final_selection_path = os.path.join(output_dir, 'hrt_final_selection.csv')
    calibration_path = os.path.join(output_dir, 'hrt_head_calibration.csv')
    summary_path = os.path.join(output_dir, 'hrt_summary.json')

    candidate_groups_df.to_csv(candidate_groups_path, index=False)
    head_references_df.to_csv(head_references_path, index=False)
    sample_scores_df.to_csv(sample_scores_path, index=False)
    group_decisions_df.to_csv(group_decisions_path, index=False)
    final_selection_df.to_csv(final_selection_path, index=False)
    calibration_df.to_csv(calibration_path, index=False)

    deviation_maps_path = None
    if save_patch_maps and deviation_maps is not None:
        deviation_maps_path = os.path.join(output_dir, 'hrt_deviation_maps.pt')
        torch.save(deviation_maps, deviation_maps_path)

    patch_bank_path = None
    if patch_bank is not None:
        patch_bank_path = os.path.join(output_dir, 'hrt_patch_bank.pt')
        torch.save(patch_bank, patch_bank_path)

    payload = {
        'summary': summary,
        'metadata': metadata,
        'artifacts': {
            'hrt_candidate_groups_csv': candidate_groups_path,
            'hrt_head_references_csv': head_references_path,
            'hrt_sample_scores_csv': sample_scores_path,
            'hrt_group_decisions_csv': group_decisions_path,
            'hrt_final_selection_csv': final_selection_path,
            'hrt_head_calibration_csv': calibration_path,
            'hrt_deviation_maps_pt': deviation_maps_path,
            'hrt_patch_bank_pt': patch_bank_path,
        },
    }
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    return {
        'hrt_candidate_groups_csv': candidate_groups_path,
        'hrt_head_references_csv': head_references_path,
        'hrt_sample_scores_csv': sample_scores_path,
        'hrt_group_decisions_csv': group_decisions_path,
        'hrt_final_selection_csv': final_selection_path,
        'hrt_head_calibration_csv': calibration_path,
        'hrt_summary_json': summary_path,
        'hrt_deviation_maps_pt': deviation_maps_path,
        'hrt_patch_bank_pt': patch_bank_path,
    }
