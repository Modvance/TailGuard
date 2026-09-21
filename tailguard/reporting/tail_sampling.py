import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from tailguard.method.tail_sampling import build_tail_sampler


def _safe_binary_metric(metric_fn, y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    mask = ~pd.isna(y_true) & ~pd.isna(y_score)
    mask = mask & np.isin(y_true, [0, 1])
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float('nan')
    return float(metric_fn(y_true.astype(int), y_score.astype(float)))


def _normalize_dataset_name(dataset_name: str) -> str:
    return str(dataset_name or '').lower().replace('_', '').replace('-', '')


def build_tail_ground_truth(metadata_df: pd.DataFrame, args) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], int, str, str]:
    metadata_df = metadata_df.copy()
    class_counts = metadata_df.groupby('class_id').size().to_dict()
    gt_class_counts = metadata_df['class_id'].map(class_counts).astype(int).to_numpy()
    gt_mode = getattr(args, 'tailsampler_gt_mode', 'dataset_rule')

    if gt_mode == 'true_class':
        threshold = int(getattr(args, 'tailsampler_tail_count_thr', 8))
        is_gt_tail = (gt_class_counts <= threshold).astype(int)
        gt_rule = f'class_count<={threshold}'
        return gt_class_counts, is_gt_tail, {int(k): int(v) for k, v in class_counts.items()}, threshold, gt_rule, 'true_class'

    if gt_mode == 'dataset_rule':
        dataset_name = str(getattr(args, 'tailsampler_dataset_name', '') or '')
        dataset_name_norm = _normalize_dataset_name(dataset_name)
        if 'stepk1' in dataset_name_norm:
            threshold = 1
            gt_rule = 'dataset_rule:step_k1'
            is_gt_tail = (gt_class_counts <= threshold).astype(int)
        elif 'stepk4' in dataset_name_norm:
            threshold = 4
            gt_rule = 'dataset_rule:step_k4'
            is_gt_tail = (gt_class_counts <= threshold).astype(int)
        elif 'pareto' in dataset_name_norm:
            threshold = 19
            gt_rule = 'dataset_rule:pareto(<20)'
            is_gt_tail = (gt_class_counts < 20).astype(int)
        else:
            raise ValueError('tailsampler_gt_mode=dataset_rule requires dataset name containing step_k1/stepk1, step_k4/stepk4, or pareto')
        return gt_class_counts, is_gt_tail, {int(k): int(v) for k, v in class_counts.items()}, threshold, gt_rule, 'paper'

    raise ValueError(f'unsupported tailsampler_gt_mode: {gt_mode}')


def compute_selection_metrics(selected: np.ndarray, is_gt_tail: np.ndarray) -> Dict[str, float]:
    selected = np.asarray(selected).astype(int)
    is_gt_tail = np.asarray(is_gt_tail).astype(int)
    tp = int(np.sum((selected == 1) & (is_gt_tail == 1)))
    fp = int(np.sum((selected == 1) & (is_gt_tail == 0)))
    tn = int(np.sum((selected == 0) & (is_gt_tail == 0)))
    fn = int(np.sum((selected == 0) & (is_gt_tail == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'selected_ratio': float(selected.mean()) if len(selected) > 0 else 0.0,
    }


def compute_ranking_metrics(tail_score: np.ndarray, is_gt_tail: np.ndarray) -> Dict[str, float]:
    return {
        'auroc': _safe_binary_metric(roc_auc_score, is_gt_tail, tail_score),
        'auprc': _safe_binary_metric(average_precision_score, is_gt_tail, tail_score),
    }


def _build_noise_proxy(metadata_df: pd.DataFrame) -> np.ndarray:
    if 'rel_path' in metadata_df.columns:
        image_names = metadata_df['rel_path'].astype(str).map(os.path.basename)
    elif 'img_path' in metadata_df.columns:
        image_names = metadata_df['img_path'].astype(str).map(os.path.basename)
    else:
        return np.zeros(len(metadata_df), dtype=int)

    image_stems = image_names.map(lambda value: os.path.splitext(value)[0])
    return np.array([0 if stem.isnumeric() else 1 for stem in image_stems], dtype=int)


def _compute_cutoff_summary(class_sizes_pred: np.ndarray, selected: np.ndarray) -> Dict[str, float]:
    selected_scores = class_sizes_pred[selected == 1]
    if selected_scores.size == 0:
        return {
            'selected_cutoff_max_pred_class_size': None,
            'selected_cutoff_min_tail_score': None,
        }
    return {
        'selected_cutoff_max_pred_class_size': float(np.max(selected_scores)),
        'selected_cutoff_min_tail_score': float(np.min(-selected_scores)),
    }


def _count_pred_class_size_le(class_sizes_pred: np.ndarray, threshold: float) -> int:
    return int(np.sum(class_sizes_pred <= float(threshold)))


def build_tail_candidate_partition(details_df: pd.DataFrame):
    if 'sample_idx' not in details_df.columns or 'is_selected' not in details_df.columns:
        raise ValueError('details_df must contain sample_idx and is_selected')
    candidate_df = details_df[details_df['is_selected'].astype(int) == 1].copy().reset_index(drop=True)
    head_df = details_df[details_df['is_selected'].astype(int) == 0].copy().reset_index(drop=True)
    if 'sample_idx' in candidate_df.columns:
        candidate_df['sample_idx'] = candidate_df['sample_idx'].astype(int)
    if 'sample_idx' in head_df.columns:
        head_df['sample_idx'] = head_df['sample_idx'].astype(int)
    return candidate_df, head_df


def run_tail_sampler_analysis(embeddings,
                              metadata_df: pd.DataFrame,
                              args):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError('embeddings must have shape [N, D]')
    if len(metadata_df) != embeddings.shape[0]:
        raise ValueError('metadata and embeddings size mismatch')

    sampler, sampler_name = build_tail_sampler(
        percentile=float(getattr(args, 'tailsampler_percentile', 0.15)),
    )

    feature_tensor = torch.from_numpy(embeddings)
    _, tail_indices, class_sizes_pred = sampler.run(feature_tensor, return_class_sizes=True)
    class_sizes_pred = class_sizes_pred.detach().cpu().numpy().astype(float)

    selected = np.zeros(len(metadata_df), dtype=int)
    selected[tail_indices.detach().cpu().numpy()] = 1
    partition_diagnostics = getattr(sampler, 'last_partition_diagnostics', None)
    if partition_diagnostics is None:
        partition_diagnostics = {
            'guard_name': None,
            'enabled': False,
            'triggered': False,
            'status': 'unavailable_for_sampler',
            'num_selected_raw': int(selected.sum()),
            'num_selected_final': int(selected.sum()),
            'num_removed_by_guard': 0,
            'selected_ratio_raw': float(selected.mean()) if len(selected) > 0 else 0.0,
            'selected_ratio_final': float(selected.mean()) if len(selected) > 0 else 0.0,
            'original_cutoff': None,
            'effective_cutoff': None,
            'num_inferred_classes': 0,
            'boundary_multiplicity': 0,
            'gap_lower': None,
            'gap_upper': None,
            'dominant_log_gap': None,
        }

    original_cutoff = partition_diagnostics.get('original_cutoff')
    if original_cutoff is None:
        selected_raw = selected.copy()
    else:
        selected_raw = (class_sizes_pred <= float(original_cutoff)).astype(int)

    gt_class_counts, is_gt_tail, class_counts, tail_max_count, gt_rule, gt_mode_name = build_tail_ground_truth(metadata_df, args)
    tail_score = -class_sizes_pred
    ranking_metrics = compute_ranking_metrics(tail_score, is_gt_tail)
    selection_metrics = compute_selection_metrics(selected, is_gt_tail)
    cutoff_summary = _compute_cutoff_summary(class_sizes_pred, selected)

    is_noise_proxy = _build_noise_proxy(metadata_df)
    selected_noise_proxy_count = int(np.sum(is_noise_proxy[selected == 1]))
    num_selected = int(selected.sum())
    selected_noise_proxy_rate = selected_noise_proxy_count / num_selected if num_selected > 0 else 0.0
    selected_gt_tail_noise_proxy_rate = (
        float(np.mean(is_noise_proxy[(selected == 1) & (is_gt_tail == 1)]))
        if np.any((selected == 1) & (is_gt_tail == 1))
        else 0.0
    )

    details_df = metadata_df.copy()
    details_df['gt_class_count'] = gt_class_counts.astype(int)
    details_df['is_gt_tail'] = is_gt_tail.astype(int)
    details_df['pred_class_size'] = class_sizes_pred.astype(float)
    details_df['tail_score'] = tail_score.astype(float)
    details_df['is_selected_raw'] = selected_raw.astype(int)
    details_df['is_selected'] = selected.astype(int)
    details_df['removed_by_partition_guard'] = (
        (selected_raw == 1) & (selected == 0)
    ).astype(int)
    details_df['is_noise_proxy'] = is_noise_proxy.astype(int)

    summary_row = {
        'run_name': getattr(args, 'save_name', 'tailsampler_run'),
        'dataset_name': getattr(args, 'tailsampler_dataset_name', os.path.basename(str(getattr(args, 'data_path', '')))),
        'sampler_name': sampler_name,
        'tailsampler_type': 'adaptive_trim_mode',
        'tail_th_type': 'trim_min',
        'vote_type': 'mode',
        'embedding_source': 'encoder_cls',
        'num_samples': int(len(details_df)),
        'num_classes': int(len(class_counts)),
        'gt_mode': gt_mode_name,
        'gt_rule': gt_rule,
        'tail_max_count': int(tail_max_count),
        'num_gt_tail_samples': int(is_gt_tail.sum()),
        'num_selected': num_selected,
        'selection_precision': float(selection_metrics['precision']),
        'selection_recall': float(selection_metrics['recall']),
        'selection_f1': float(selection_metrics['f1']),
        'selection_tp': int(selection_metrics['tp']),
        'selection_fp': int(selection_metrics['fp']),
        'selection_tn': int(selection_metrics['tn']),
        'selection_fn': int(selection_metrics['fn']),
        'selected_ratio': float(selection_metrics['selected_ratio']),
        'tail_auroc': float(ranking_metrics['auroc']) if not np.isnan(ranking_metrics['auroc']) else None,
        'tail_auprc': float(ranking_metrics['auprc']) if not np.isnan(ranking_metrics['auprc']) else None,
        'selected_noise_proxy_count': selected_noise_proxy_count,
        'selected_noise_proxy_rate': float(selected_noise_proxy_rate),
        'selected_gt_tail_noise_proxy_rate': float(selected_gt_tail_noise_proxy_rate),
        'selected_cutoff_max_pred_class_size': cutoff_summary['selected_cutoff_max_pred_class_size'],
        'selected_cutoff_min_tail_score': cutoff_summary['selected_cutoff_min_tail_score'],
        'partition_guard_name': partition_diagnostics.get('guard_name'),
        'partition_guard_enabled': bool(partition_diagnostics.get('enabled', False)),
        'partition_guard_triggered': bool(partition_diagnostics.get('triggered', False)),
        'partition_guard_status': partition_diagnostics.get('status'),
        'partition_guard_num_selected_raw': int(partition_diagnostics.get('num_selected_raw', num_selected)),
        'partition_guard_num_selected_final': int(partition_diagnostics.get('num_selected_final', num_selected)),
        'partition_guard_num_removed': int(partition_diagnostics.get('num_removed_by_guard', 0)),
        'partition_guard_selected_ratio_raw': float(partition_diagnostics.get('selected_ratio_raw', selection_metrics['selected_ratio'])),
        'partition_guard_selected_ratio_final': float(partition_diagnostics.get('selected_ratio_final', selection_metrics['selected_ratio'])),
        'partition_guard_original_cutoff': partition_diagnostics.get('original_cutoff'),
        'partition_guard_effective_cutoff': partition_diagnostics.get('effective_cutoff'),
        'partition_guard_num_inferred_classes': int(partition_diagnostics.get('num_inferred_classes', 0)),
        'partition_guard_boundary_multiplicity': int(partition_diagnostics.get('boundary_multiplicity', 0)),
        'partition_guard_gap_lower': partition_diagnostics.get('gap_lower'),
        'partition_guard_gap_upper': partition_diagnostics.get('gap_upper'),
        'partition_guard_dominant_log_gap': partition_diagnostics.get('dominant_log_gap'),
        'num_pred_class_size_le_1': _count_pred_class_size_le(class_sizes_pred, 1),
        'num_pred_class_size_le_5': _count_pred_class_size_le(class_sizes_pred, 5),
        'num_pred_class_size_le_10': _count_pred_class_size_le(class_sizes_pred, 10),
        'num_pred_class_size_le_20': _count_pred_class_size_le(class_sizes_pred, 20),
        'mean_pred_class_size': float(class_sizes_pred.mean()) if len(class_sizes_pred) > 0 else None,
        'mean_pred_class_size_tail': float(class_sizes_pred[is_gt_tail == 1].mean()) if np.any(is_gt_tail == 1) else None,
        'mean_pred_class_size_head': float(class_sizes_pred[is_gt_tail == 0].mean()) if np.any(is_gt_tail == 0) else None,
    }

    return summary_row, details_df


def save_tail_sampler_artifacts(output_dir: str, summary_row: Dict, details_df: pd.DataFrame, metadata: Dict, save_details: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    details_csv_path = os.path.join(output_dir, 'sampler_analysis_details.csv')
    if save_details:
        details_df.to_csv(details_csv_path, index=False)

    summary_json_path = os.path.join(output_dir, 'sampler_analysis_summary.json')
    payload = {
        'summary': summary_row,
        'metadata': metadata,
        'artifacts': {
            'sampler_analysis_details_csv': details_csv_path if save_details else None,
        },
    }
    with open(summary_json_path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    return {
        'sampler_analysis_details_csv': details_csv_path if save_details else None,
        'sampler_analysis_summary_json': summary_json_path,
    }
