"""Artifact writers for the canonical TailGuard method."""

import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch


def _make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path: str, payload: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(_make_json_safe(payload), file, indent=2, ensure_ascii=False)



def save_tailguard_prepare_artifacts(output_dir: str,
                                       train_metadata_df: pd.DataFrame,
                                       metadata: Dict,
                                       analysis_metadata_df: Optional[pd.DataFrame] = None):
    os.makedirs(output_dir, exist_ok=True)
    train_metadata_path = os.path.join(output_dir, 'tailguard_train_metadata.csv')
    analysis_metadata_path = os.path.join(output_dir, 'tailguard_train_analysis_metadata.csv')
    summary_path = os.path.join(output_dir, 'prepare_summary.json')

    train_metadata_df.to_csv(train_metadata_path, index=False)
    if analysis_metadata_df is not None:
        analysis_metadata_df.to_csv(analysis_metadata_path, index=False)
    else:
        analysis_metadata_path = None

    artifacts = {
        'tailguard_train_metadata_csv': train_metadata_path,
        'tailguard_train_analysis_metadata_csv': analysis_metadata_path,
    }
    _write_json(summary_path, {'metadata': metadata, 'artifacts': artifacts})
    artifacts['prepare_summary_json'] = summary_path
    return artifacts


def save_tailguard_gbps_iteration_artifacts(iter_dir: str,
                                             train_scores_df: Optional[pd.DataFrame],
                                             h_group_metrics_df: pd.DataFrame,
                                             summary: Dict):
    os.makedirs(iter_dir, exist_ok=True)
    train_scores_path = os.path.join(iter_dir, 'train_scores.csv')
    h_group_metrics_path = os.path.join(iter_dir, 'h_group_metrics.csv')
    summary_path = os.path.join(iter_dir, 'summary.json')

    if train_scores_df is not None:
        train_scores_df.to_csv(train_scores_path, index=False)
    else:
        train_scores_path = None
    h_group_metrics_df.to_csv(h_group_metrics_path, index=False)
    _write_json(summary_path, summary)
    return {
        'train_scores_csv': train_scores_path,
        'h_group_metrics_csv': h_group_metrics_path,
        'summary_json': summary_path,
    }


def save_tailguard_trigger_summary(output_dir: str, summary: Dict):
    path = os.path.join(output_dir, 'gbps_trigger_summary.json')
    _write_json(path, summary)
    return path


def save_tailguard_head_prune_artifacts(output_dir: str,
                                          decisions_df: pd.DataFrame,
                                          group_summary_df: pd.DataFrame,
                                          summary: Dict):
    os.makedirs(output_dir, exist_ok=True)
    decisions_path = os.path.join(output_dir, 'h_prune_decisions.csv')
    group_summary_path = os.path.join(output_dir, 'h_prune_group_summary.csv')
    summary_path = os.path.join(output_dir, 'h_prune_summary.json')
    decisions_df.to_csv(decisions_path, index=False)
    group_summary_df.to_csv(group_summary_path, index=False)
    _write_json(summary_path, summary)
    return {
        'h_prune_decisions_csv': decisions_path,
        'h_prune_group_summary_csv': group_summary_path,
        'h_prune_summary_json': summary_path,
    }


def save_tailguard_attachment_artifacts(output_dir: str,
                                          geometry: Dict,
                                          reconciliation_scores_df: pd.DataFrame,
                                          reconciliation_summary: Dict,
                                          tail_open_df: pd.DataFrame,
                                          tail_head_normal_df: pd.DataFrame,
                                          rgd_bic_candidates_df: Optional[pd.DataFrame] = None):
    """Save one canonical record for each TRP result.

    The RGD implementation has one score table and one set reassigned to the
    head structure.  The public artifacts mirror those method-level concepts
    instead of exposing internal aliases.
    """
    os.makedirs(output_dir, exist_ok=True)
    geometry_path = os.path.join(output_dir, 'clean_head_group_geometry.pt')
    reconciliation_scores_path = os.path.join(output_dir, 'tail_reconciliation_scores.csv')
    reconciliation_summary_path = os.path.join(output_dir, 'tail_reconciliation_summary.json')
    tail_open_path = os.path.join(output_dir, 'tail_open_samples.csv')
    tail_head_normal_path = os.path.join(output_dir, 'tail_head_normal_samples.csv')
    rgd_bic_candidates_path = os.path.join(output_dir, 'rgd_segmented_bic_candidates.csv')

    torch.save(geometry, geometry_path)
    reconciliation_scores_df.to_csv(reconciliation_scores_path, index=False)
    tail_open_df.to_csv(tail_open_path, index=False)
    tail_head_normal_df.to_csv(tail_head_normal_path, index=False)
    _write_json(reconciliation_summary_path, reconciliation_summary)
    if rgd_bic_candidates_df is not None and len(rgd_bic_candidates_df) > 0:
        rgd_bic_candidates_df.to_csv(rgd_bic_candidates_path, index=False)
    else:
        rgd_bic_candidates_path = None
    return {
        'clean_head_group_geometry_pt': geometry_path,
        'tail_reconciliation_scores_csv': reconciliation_scores_path,
        'tail_reconciliation_summary_json': reconciliation_summary_path,
        'tail_open_samples_csv': tail_open_path,
        'tail_head_normal_samples_csv': tail_head_normal_path,
        'rgd_segmented_bic_candidates_csv': rgd_bic_candidates_path,
    }


def save_tailguard_stage2_artifacts(output_dir: str,
                                      retained_samples_df: pd.DataFrame,
                                      removed_samples_df: pd.DataFrame,
                                      summary: Dict):
    os.makedirs(output_dir, exist_ok=True)
    retained_path = os.path.join(output_dir, 'stage2_retained_samples.csv')
    removed_path = os.path.join(output_dir, 'stage2_removed_samples.csv')
    summary_path = os.path.join(output_dir, 'stage2_dataset_summary.json')
    retained_samples_df.to_csv(retained_path, index=False)
    removed_samples_df.to_csv(removed_path, index=False)
    _write_json(summary_path, summary)
    return {
        'stage2_retained_samples_csv': retained_path,
        'stage2_removed_samples_csv': removed_path,
        'stage2_dataset_summary_json': summary_path,
    }


def save_tailguard_pseudoclass_artifacts(output_dir: str,
                                           members_df: pd.DataFrame,
                                           classes_df: pd.DataFrame,
                                           tail_edges_df: pd.DataFrame,
                                           summary: Dict,
                                           metadata: Dict = None):
    os.makedirs(output_dir, exist_ok=True)
    members_path = os.path.join(output_dir, 'pseudo_class_members.csv')
    classes_path = os.path.join(output_dir, 'pseudo_classes.csv')
    tail_edges_path = os.path.join(output_dir, 'tail_adaptive_neighborhood_edges.csv')
    summary_path = os.path.join(output_dir, 'pseudo_class_registry.json')
    members_df.to_csv(members_path, index=False)
    classes_df.to_csv(classes_path, index=False)
    tail_edges_df.to_csv(tail_edges_path, index=False)
    registry = dict(summary)
    if metadata is not None:
        registry.update(metadata)
    _write_json(summary_path, registry)
    return {
        'pseudo_class_members_csv': members_path,
        'pseudo_classes_csv': classes_path,
        'tail_adaptive_neighborhood_edges_csv': tail_edges_path,
        'pseudo_class_registry_json': summary_path,
    }


def save_tailguard_pseudoclass_report_artifacts(output_dir: str,
                                                  summary: Dict,
                                                  predictions_df: pd.DataFrame,
                                                  class_summary_df: pd.DataFrame,
                                                  contingency_df: pd.DataFrame):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'stage1_cleanup_pseudoclass_report.json')
    predictions_path = os.path.join(output_dir, 'stage1_cleanup_pseudoclass_predictions.csv')
    class_summary_path = os.path.join(output_dir, 'stage1_cleanup_pseudoclass_summary.csv')
    contingency_path = os.path.join(output_dir, 'stage1_cleanup_pseudoclass_contingency.csv')
    _write_json(summary_path, summary)
    predictions_df.to_csv(predictions_path, index=False)
    class_summary_df.to_csv(class_summary_path, index=False)
    contingency_df.to_csv(contingency_path, index=False)
    return {
        'stage1_cleanup_pseudoclass_report_json': summary_path,
        'stage1_cleanup_pseudoclass_predictions_csv': predictions_path,
        'stage1_cleanup_pseudoclass_summary_csv': class_summary_path,
        'stage1_cleanup_pseudoclass_contingency_csv': contingency_path,
    }


def save_tailguard_denoising_diagnostics(output_dir: str,
                                            summary: Dict,
                                            class_rows_df: pd.DataFrame):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'denoising_diagnostics.json')
    class_rows_path = os.path.join(output_dir, 'denoising_diagnostics_by_class.csv')
    _write_json(summary_path, summary)
    class_rows_df.to_csv(class_rows_path, index=False)
    return {
        'denoising_diagnostics_json': summary_path,
        'denoising_diagnostics_by_class_csv': class_rows_path,
    }


def save_tailguard_memory_artifacts(output_dir: str,
                                      memory_system: Dict,
                                      score_df: pd.DataFrame,
                                      per_class_metrics,
                                      summary: Dict,
                                      metadata: Dict,
                                      metrics_by_mode: Optional[Dict] = None):
    os.makedirs(output_dir, exist_ok=True)
    system_path = os.path.join(output_dir, 'pseudo_class_memory_system.pt')
    class_summary_path = os.path.join(output_dir, 'pseudo_class_memory_summary.csv')
    score_path = os.path.join(output_dir, 'memory_eval_scores.csv')
    summary_path = os.path.join(output_dir, 'memory_eval_summary.json')

    torch.save(memory_system, system_path)
    rows = []
    for pseudo_class_id, entry in sorted(memory_system['class_entries'].items(), key=lambda item: int(item[0])):
        spatial_size = entry.get('spatial_size') or (0, 0)
        rows.append({
            'pseudo_class_id': int(pseudo_class_id),
            'pseudo_class_type': str(entry['pseudo_class_type']),
            'num_members': int(entry['num_members']),
            'has_memory_bank': bool(entry['has_memory_bank']),
            'num_patches': int(entry['num_patches']),
            'feature_dim': int(entry['feature_dim']),
            'spatial_h': int(spatial_size[0]),
            'spatial_w': int(spatial_size[1]),
        })
    pd.DataFrame(rows).to_csv(class_summary_path, index=False)
    score_df.to_csv(score_path, index=False)

    payload = {
        'summary': summary,
        'per_class_metrics': per_class_metrics,
        'metadata': metadata,
    }
    if metrics_by_mode is not None:
        payload['metrics_by_mode'] = metrics_by_mode
    _write_json(summary_path, payload)
    return {
        'pseudo_class_memory_system_pt': system_path,
        'pseudo_class_memory_summary_csv': class_summary_path,
        'memory_eval_scores_csv': score_path,
        'memory_eval_summary_json': summary_path,
    }


def save_tailguard_summary(output_dir: str, summary: Dict):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'tailguard_summary.json')
    _write_json(summary_path, summary)
    return summary_path
