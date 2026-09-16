"""H-only GBPS, checkpoint, cleanup, and Stage-2 planning for TailGuard."""

import math
import os
import random
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from tailguard.method.dhp_utils import build_gbps_hard_delete_plan, summarize_contamination_labels
from tailguard.config import TAILGUARD_FINAL_DEFAULTS


def _iter_number_from_name(name: str):
    match = re.match(r'^iter_(\d+)$', str(name))
    if match is None:
        return None
    return int(match.group(1))


def _build_retained_index_map(samples_df: pd.DataFrame, all_samples_df: Optional[pd.DataFrame] = None):
    retained_index_map = {}
    if all_samples_df is not None and 'class_id' in all_samples_df.columns:
        for class_id in sorted(all_samples_df['class_id'].astype(int).unique().tolist()):
            retained_index_map[int(class_id)] = []
    if len(samples_df) == 0:
        return retained_index_map
    for class_id, class_df in samples_df.groupby('class_id', sort=True):
        ordered = class_df.sort_values(['base_idx', 'sample_idx'], kind='mergesort')
        retained_index_map[int(class_id)] = [int(value) for value in ordered['base_idx'].astype(int).tolist()]
    return retained_index_map


def _metadata_view(metadata_df: pd.DataFrame):
    columns = [
        'sample_idx', 'sample_key', 'img_path', 'base_idx', 'tail_candidate', 'pred_class_size',
        'tail_score', 'adaptive_angle', 'group_id', 'group_size',
    ]
    view = metadata_df[columns].copy()
    view['sample_idx'] = view['sample_idx'].astype(int)
    view['tail_candidate'] = view['tail_candidate'].astype(int)
    view['group_id'] = view['group_id'].astype(int)
    return view


def enrich_scores_with_tailguard_metadata(scored_df: pd.DataFrame, metadata_df: pd.DataFrame):
    scored = scored_df.drop(columns=['is_contaminated', 'class_name'], errors='ignore').copy()
    scored['sample_idx'] = scored['sample_idx'].astype(int)
    metadata_view = _metadata_view(metadata_df)
    drop_columns = [column for column in metadata_view.columns if column in scored.columns and column != 'sample_idx']
    scored = scored.drop(columns=drop_columns, errors='ignore')
    scored = scored.merge(metadata_view, on='sample_idx', how='left', validate='one_to_one')
    if scored['tail_candidate'].isna().any():
        missing_count = int(scored['tail_candidate'].isna().sum())
        raise ValueError('missing TailGuard metadata for {} scored samples'.format(missing_count))
    scored['tail_candidate'] = scored['tail_candidate'].astype(int)
    scored['group_id'] = scored['group_id'].astype(int)
    return scored


def run_tailguard_gbps_iteration(scored_df: pd.DataFrame,
                                   head_group_assignments_df: pd.DataFrame,
                                   metadata_df: pd.DataFrame,
                                   args):
    train_scores_df = enrich_scores_with_tailguard_metadata(scored_df, metadata_df)
    h_scores_df = train_scores_df.loc[train_scores_df['tail_candidate'] == 0].copy().reset_index(drop=True)
    if len(h_scores_df) == 0:
        raise ValueError('TailGuard GBPS requires non-empty H scores')

    h_scores_for_gbps_df = h_scores_df.drop(columns=['group_id', 'group_size'], errors='ignore')
    from tailguard.method.dhp_statistics import compute_gbps_u_with_bootstrap
    gbps_result = compute_gbps_u_with_bootstrap(
        h_scores_for_gbps_df.copy(),
        head_group_assignments_df.copy(),
        B=int(getattr(args, 'gbps_bootstrap_B', 20)),
        args=args,
    )
    summary = dict(gbps_result['summary'])
    summary.update({
        'tailguard_h_only': True,
        'num_scored_total': int(len(train_scores_df)),
        'num_scored_h': int(len(h_scores_df)),
        'num_scored_t_saved_only': int((train_scores_df['tail_candidate'] == 1).sum()),
    })
    return {
        'gbps_U': gbps_result['gbps_U'],
        'gbps_SE': gbps_result['gbps_SE'],
        'global_noise_evidence': gbps_result['global_noise_evidence'],
        'train_scores_df': train_scores_df,
        'h_scores_df': h_scores_df,
        'h_group_metrics_df': gbps_result['group_metrics_df'],
        'h_sample_group_scores_df': gbps_result['sample_group_scores_df'],
        'bootstrap_df': gbps_result['bootstrap_df'],
        'summary': summary,
    }


def save_tailguard_selected_checkpoint(path: str,
                                         model,
                                         optimizer,
                                         scheduler,
                                         iteration: int,
                                         gbps_U: float,
                                         gbps_SE: float,
                                         args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'iteration': int(iteration),
        'gbps_U': float(gbps_U),
        'gbps_SE': float(gbps_SE),
        'rng_state': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        },
        'args': vars(args).copy(),
    }
    torch.save(payload, path)
    return path


def load_tailguard_selected_checkpoint(path: str, model, optimizer=None, scheduler=None, map_location=None, restore_rng: bool = True):
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if restore_rng and checkpoint.get('rng_state') is not None:
        rng_state = checkpoint['rng_state']
        if rng_state.get('torch') is not None:
            torch_state = rng_state['torch']
            if torch.is_tensor(torch_state):
                torch_state = torch_state.detach().cpu().byte()
            torch.set_rng_state(torch_state)
        if torch.cuda.is_available() and rng_state.get('cuda') is not None:
            cuda_states = []
            for cuda_state in rng_state['cuda']:
                if torch.is_tensor(cuda_state):
                    cuda_state = cuda_state.detach().cpu().byte()
                cuda_states.append(cuda_state)
            torch.cuda.set_rng_state_all(cuda_states)
        if rng_state.get('numpy') is not None:
            np.random.set_state(rng_state['numpy'])
        if rng_state.get('python') is not None:
            random.setstate(rng_state['python'])
    return checkpoint


def _coerce_bool(value, default=False):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'true', '1', 'yes'}:
            return True
        if normalized in {'false', '0', 'no', ''}:
            return False
    return bool(value)


def _merge_scores_with_head_groups(scored_df: pd.DataFrame, group_assignments_df: pd.DataFrame):
    if 'sample_idx' not in scored_df.columns:
        raise ValueError('head pruning requires sample_idx in scored scores')
    if 'sample_key' not in group_assignments_df.columns or 'group_id' not in group_assignments_df.columns:
        raise ValueError('head group assignments must contain sample_key and group_id')

    assignments = group_assignments_df[['sample_key', 'group_id']].copy()
    assignments['sample_key'] = assignments['sample_key'].astype(int)
    assignments['group_id'] = assignments['group_id'].astype(int)
    conflicting = assignments.groupby('sample_key')['group_id'].nunique()
    if bool((conflicting > 1).any()):
        raise ValueError('head group assignments contain conflicting group_id values')
    assignments = assignments.drop_duplicates(subset=['sample_key'], keep='first')

    merged = scored_df.drop(columns=['sample_key', 'group_id'], errors='ignore').copy()
    merged['sample_idx'] = merged['sample_idx'].astype(int)
    merged = merged.merge(
        assignments,
        left_on='sample_idx',
        right_on='sample_key',
        how='left',
        validate='one_to_one',
    )
    if merged['group_id'].isna().any():
        missing_count = int(merged['group_id'].isna().sum())
        raise ValueError('missing head group assignment for {} scored samples'.format(missing_count))
    merged['group_id'] = merged['group_id'].astype(int)
    observed_group_sizes = merged.groupby('group_id')['sample_idx'].transform('size').astype(int)
    if 'group_size' in merged.columns:
        declared_group_sizes = pd.to_numeric(merged['group_size'], errors='coerce')
        if declared_group_sizes.isna().any():
            raise ValueError('head pruning scores contain invalid group_size values')
        if bool((declared_group_sizes.astype(int) != observed_group_sizes).any()):
            raise ValueError('head pruning scores contain group_size values inconsistent with assignments')
    merged['group_size'] = observed_group_sizes
    for column in ['class_id', 'base_idx']:
        if column not in merged.columns:
            raise ValueError('head pruning requires {} in scored scores'.format(column))
        merged[column] = merged[column].astype(int)
    if 'image_score' not in merged.columns:
        raise ValueError('head pruning requires image_score in scored scores')
    return merged


def _head_metric_window_iters(gbps_dir: str, selected_iter: int, stable_window: int):
    if not os.path.isdir(gbps_dir):
        raise FileNotFoundError('TailGuard GBPS directory does not exist: {}'.format(gbps_dir))
    iter_numbers = []
    for name in os.listdir(gbps_dir):
        iter_number = _iter_number_from_name(name)
        if iter_number is None or int(iter_number) > int(selected_iter):
            continue
        metrics_path = os.path.join(gbps_dir, name, 'h_group_metrics.csv')
        if os.path.isfile(metrics_path):
            iter_numbers.append(int(iter_number))
    iter_numbers = sorted(iter_numbers)
    return iter_numbers[-max(1, int(stable_window)):]


def load_tailguard_head_prune_group_context(gbps_dir: str, selected_iter: int, args):
    """Load selected GMM mass and temporal active-gate evidence for H groups."""
    stable_window = int(getattr(
        args,
        'gbps_prune_stable_window',
        TAILGUARD_FINAL_DEFAULTS['gbps_prune_stable_window'],
    ))
    min_observations = int(getattr(
        args,
        'gbps_prune_stable_min_observations',
        TAILGUARD_FINAL_DEFAULTS['gbps_prune_stable_min_observations'],
    ))
    min_active_ratio = float(getattr(
        args,
        'gbps_prune_min_active_ratio',
        TAILGUARD_FINAL_DEFAULTS['gbps_prune_min_active_ratio'],
    ))
    if stable_window < 1:
        raise ValueError('gbps_prune_stable_window must be at least 1')
    if min_observations < 1:
        raise ValueError('gbps_prune_stable_min_observations must be at least 1')
    if not 0.0 <= min_active_ratio < 1.0:
        raise ValueError('gbps_prune_min_active_ratio must be in [0, 1)')

    selected_iter = int(selected_iter)
    selected_metrics_path = os.path.join(gbps_dir, 'iter_{:05d}'.format(selected_iter), 'h_group_metrics.csv')
    if not os.path.isfile(selected_metrics_path):
        raise FileNotFoundError(
            'adaptive H pruning requires selected h_group_metrics.csv: {}'.format(selected_metrics_path)
        )
    selected_metrics_df = pd.read_csv(selected_metrics_path)
    required = ['group_id', 'pi_high', 'is_active_group']
    missing = [column for column in required if column not in selected_metrics_df.columns]
    if missing:
        raise ValueError('selected H group metrics missing columns: {}'.format(', '.join(missing)))
    selected_metrics_df['group_id'] = selected_metrics_df['group_id'].astype(int)
    if selected_metrics_df['group_id'].duplicated().any():
        raise ValueError('selected H group metrics contain duplicate group_id values')

    window_iters = _head_metric_window_iters(gbps_dir, selected_iter, stable_window)
    observations = {}
    for iter_number in window_iters:
        metrics_path = os.path.join(gbps_dir, 'iter_{:05d}'.format(iter_number), 'h_group_metrics.csv')
        metrics_df = pd.read_csv(metrics_path)
        if 'group_id' not in metrics_df.columns:
            continue
        metrics_df['group_id'] = metrics_df['group_id'].astype(int)
        if metrics_df['group_id'].duplicated().any():
            raise ValueError('H group metrics contain duplicate group_id values: {}'.format(metrics_path))
        for _, row in metrics_df.iterrows():
            group_id = int(row['group_id'])
            is_valid = _coerce_bool(row.get('valid_group', True), default=True)
            if not is_valid:
                continue
            is_active = _coerce_bool(
                row.get('is_active_group', float(row.get('q_g', 0.0)) > 0.0),
                default=False,
            )
            observations.setdefault(group_id, []).append((int(iter_number), bool(is_active)))

    context_rows = []
    for _, selected_row in selected_metrics_df.sort_values('group_id').iterrows():
        group_id = int(selected_row['group_id'])
        group_observations = observations.get(group_id, [])
        active_ratio = (
            float(np.mean([1.0 if active else 0.0 for _, active in group_observations]))
            if group_observations else 0.0
        )
        selected_valid = _coerce_bool(selected_row.get('valid_group', True), default=True)
        selected_active = selected_valid and _coerce_bool(
            selected_row.get('is_active_group', float(selected_row.get('q_g', 0.0)) > 0.0),
            default=False,
        )
        stable_active = bool(
            len(group_observations) >= min_observations
            and selected_active
            and active_ratio > min_active_ratio
        )
        context_rows.append({
            'group_id': group_id,
            'selected_group_size': int(selected_row['group_size']) if not pd.isna(selected_row.get('group_size')) else None,
            'selected_group_valid': bool(selected_valid),
            'selected_group_active': bool(selected_active),
            'selected_pi_high': float(selected_row['pi_high']) if not pd.isna(selected_row['pi_high']) else np.nan,
            'stable_group_active': stable_active,
            'stable_active_ratio': active_ratio,
            'stable_valid_observations': int(len(group_observations)),
            'stable_source_iters': ','.join(str(iter_number) for iter_number, _ in group_observations),
            'selected_metrics_iter': selected_iter,
            'selected_delta_bic': selected_row.get('delta_bic'),
            'selected_sep': selected_row.get('sep'),
            'selected_conf': selected_row.get('conf'),
        })
    return pd.DataFrame(context_rows)


def _class_retained_counts(kept_samples: pd.DataFrame):
    counts = {}
    if 'class_name' not in kept_samples.columns:
        return counts
    for class_id, class_samples in kept_samples.groupby('class_id', sort=True):
        class_name = str(class_samples['class_name'].iloc[0]) if len(class_samples) > 0 else str(class_id)
        counts[class_name] = int(len(class_samples))
    return counts


def _build_adaptive_capped_head_delete_plan(scored_df: pd.DataFrame,
                                               group_assignments_df: pd.DataFrame,
                                               group_context_df: pd.DataFrame,
                                               max_prune_ratio: float,
                                               min_keep_per_group: int):
    max_prune_ratio = float(max_prune_ratio)
    min_keep_per_group = int(min_keep_per_group)
    if not 0.0 <= max_prune_ratio <= 1.0:
        raise ValueError('gbps_prune_max_ratio must be in [0, 1]')
    if min_keep_per_group < 0:
        raise ValueError('gbps_min_keep_per_group must be non-negative')
    if 'group_id' not in group_context_df.columns:
        raise ValueError('adaptive H pruning requires group context with group_id')

    merged_df = _merge_scores_with_head_groups(scored_df, group_assignments_df)
    context = group_context_df.copy()
    context['group_id'] = context['group_id'].astype(int)
    if context['group_id'].duplicated().any():
        raise ValueError('adaptive H pruning context contains duplicate group_id values')
    score_group_ids = set(merged_df['group_id'].astype(int).unique().tolist())
    context_group_ids = set(context['group_id'].astype(int).unique().tolist())
    if score_group_ids != context_group_ids:
        raise ValueError(
            'adaptive H pruning group mismatch: missing_metrics={}, extra_metrics={}'.format(
                sorted(score_group_ids - context_group_ids),
                sorted(context_group_ids - score_group_ids),
            )
        )
    context_by_group = context.set_index('group_id', drop=False)
    decision_frames = []
    group_rows = []

    for group_id, group_samples in merged_df.groupby('group_id', sort=True):
        group_samples = group_samples.sort_values(
            ['image_score', 'sample_idx'], ascending=[False, True]
        ).reset_index(drop=True)
        group_size = int(len(group_samples))
        if int(group_id) in context_by_group.index:
            context_row = context_by_group.loc[int(group_id)]
            if isinstance(context_row, pd.DataFrame):
                context_row = context_row.iloc[0]
            selected_pi_high = pd.to_numeric(context_row.get('selected_pi_high'), errors='coerce')
            selected_pi_high = float(selected_pi_high) if not pd.isna(selected_pi_high) else np.nan
            stable_group_active = _coerce_bool(context_row.get('stable_group_active'), default=False)
            selected_group_active = _coerce_bool(context_row.get('selected_group_active'), default=False)
            selected_group_valid = _coerce_bool(context_row.get('selected_group_valid'), default=False)
            stable_active_ratio = float(context_row.get('stable_active_ratio', 0.0))
            stable_valid_observations = int(context_row.get('stable_valid_observations', 0))
            stable_source_iters = str(context_row.get('stable_source_iters', ''))
            selected_metrics_iter = context_row.get('selected_metrics_iter')
            selected_group_size = context_row.get('selected_group_size')
            selected_delta_bic = context_row.get('selected_delta_bic')
            selected_sep = context_row.get('selected_sep')
            selected_conf = context_row.get('selected_conf')
        else:
            selected_pi_high = np.nan
            stable_group_active = False
            selected_group_active = False
            selected_group_valid = False
            stable_active_ratio = 0.0
            stable_valid_observations = 0
            stable_source_iters = ''
            selected_metrics_iter = None
            selected_group_size = None
            selected_delta_bic = None
            selected_sep = None
            selected_conf = None

        if selected_group_size is not None and not pd.isna(selected_group_size):
            if int(selected_group_size) != group_size:
                raise ValueError(
                    'selected H group size mismatch for group {}: metrics={}, scores={}'.format(
                        int(group_id), int(selected_group_size), group_size
                    )
                )
        if stable_group_active and (
            np.isnan(selected_pi_high)
            or not np.isfinite(selected_pi_high)
            or not 0.0 <= selected_pi_high <= 1.0
        ):
            raise ValueError('active H group {} has invalid pi_high: {}'.format(int(group_id), selected_pi_high))

        estimated_high_ratio = (
            min(max_prune_ratio, max(0.0, min(1.0, selected_pi_high)))
            if not np.isnan(selected_pi_high) else 0.0
        )
        applied_prune_ratio = float(estimated_high_ratio) if stable_group_active else 0.0
        requested_remove = int(math.floor(group_size * applied_prune_ratio))
        max_removable = max(0, group_size - min_keep_per_group)
        remove_count = min(requested_remove, max_removable)
        effective_ratio = float(remove_count / group_size) if group_size > 0 else 0.0
        if np.isnan(selected_pi_high):
            group_reason = 'missing_selected_group_metrics'
        elif not selected_group_active:
            group_reason = 'selected_group_inactive'
        elif not stable_group_active:
            group_reason = 'unstable_active_gate'
        elif requested_remove <= 0:
            group_reason = 'adaptive_ratio_below_one_sample'
        elif remove_count < requested_remove:
            group_reason = 'min_keep_cap_applied'
        else:
            group_reason = 'adaptive_prune_applied'

        group_samples['prune_mode'] = 'adaptive'
        group_samples['prune_rank'] = np.arange(1, group_size + 1, dtype=int)
        group_samples['prune_selected'] = group_samples['prune_rank'] <= int(remove_count)
        group_samples['prune_decision'] = np.where(group_samples['prune_selected'], 'removed', 'retained')
        group_samples['prune_group_reason'] = group_reason
        group_samples['prune_group_pi_high'] = selected_pi_high
        group_samples['prune_group_selected_active'] = bool(selected_group_active)
        group_samples['prune_group_stable_active'] = bool(stable_group_active)
        group_samples['prune_group_active_ratio'] = float(stable_active_ratio)
        group_samples['prune_group_valid_observations'] = int(stable_valid_observations)
        group_samples['prune_group_source_iters'] = stable_source_iters
        group_samples['prune_estimated_high_ratio'] = float(estimated_high_ratio)
        group_samples['prune_requested_ratio'] = float(applied_prune_ratio)
        group_samples['prune_effective_ratio'] = float(effective_ratio)
        group_samples['prune_max_ratio'] = float(max_prune_ratio)
        decision_frames.append(group_samples)

        group_rows.append({
            'group_id': int(group_id),
            'group_size': group_size,
            'selected_metrics_iter': selected_metrics_iter,
            'selected_pi_high': selected_pi_high,
            'selected_group_valid': bool(selected_group_valid),
            'selected_group_active': bool(selected_group_active),
            'selected_delta_bic': selected_delta_bic,
            'selected_sep': selected_sep,
            'selected_conf': selected_conf,
            'stable_group_active': bool(stable_group_active),
            'stable_active_ratio': float(stable_active_ratio),
            'stable_valid_observations': int(stable_valid_observations),
            'stable_source_iters': stable_source_iters,
            'configured_max_ratio': float(max_prune_ratio),
            'adaptive_estimated_high_ratio': float(estimated_high_ratio),
            'adaptive_requested_ratio': float(applied_prune_ratio),
            'estimate_exceeds_cap': bool(not np.isnan(selected_pi_high) and selected_pi_high > max_prune_ratio),
            'cap_applied': bool(
                stable_group_active
                and not np.isnan(selected_pi_high)
                and selected_pi_high > max_prune_ratio
            ),
            'requested_remove': int(requested_remove),
            'max_removable': int(max_removable),
            'removed': int(remove_count),
            'retained': int(group_size - remove_count),
            'effective_prune_ratio': float(effective_ratio),
            'decision_reason': group_reason,
        })

    decisions_df = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    if len(decisions_df) > 0:
        decisions_df = decisions_df.sort_values('sample_idx').reset_index(drop=True)
    group_summary_df = pd.DataFrame(group_rows)
    pruned_samples = decisions_df.loc[decisions_df['prune_selected']].copy().reset_index(drop=True)
    kept_samples = decisions_df.loc[~decisions_df['prune_selected']].copy().reset_index(drop=True)
    group_counts = {
        str(int(row['group_id'])): row.to_dict()
        for _, row in group_summary_df.iterrows()
    }
    summary = {
        'mode': 'remove',
        'prune_mode': 'adaptive',
        'num_samples_before_prune': int(len(decisions_df)),
        'num_pruned': int(len(pruned_samples)),
        'num_retained': int(len(kept_samples)),
        'gbps_prune_max_ratio': float(max_prune_ratio),
        'gbps_min_keep_per_group': int(min_keep_per_group),
        'effective_prune_ratio': float(len(pruned_samples) / len(decisions_df)) if len(decisions_df) > 0 else 0.0,
        'num_groups': int(len(group_summary_df)),
        'num_stable_active_groups': int(group_summary_df['stable_group_active'].sum()) if len(group_summary_df) > 0 else 0,
        'group_counts': group_counts,
        'class_retained_counts': _class_retained_counts(kept_samples),
    }
    summary.update(summarize_contamination_labels(pruned_samples, prefix='pruned'))
    summary.update(summarize_contamination_labels(kept_samples, prefix='kept'))
    return {
        'mode': 'remove',
        'merged_scores_df': decisions_df,
        'prune_decisions_df': decisions_df,
        'group_summary_df': group_summary_df,
        'pruned_samples': pruned_samples,
        'kept_samples': kept_samples,
        'retained_index_map': _build_retained_index_map(kept_samples),
        'summary': summary,
    }


def _build_fixed_head_delete_plan(scored_df: pd.DataFrame,
                                    group_assignments_df: pd.DataFrame,
                                    prune_ratio: float,
                                    min_keep_per_group: int):
    if not 0.0 <= float(prune_ratio) <= 1.0:
        raise ValueError('gbps_prune_ratio must be in [0, 1]')
    if int(min_keep_per_group) < 0:
        raise ValueError('gbps_min_keep_per_group must be non-negative')
    baseline_plan = build_gbps_hard_delete_plan(
        scored_df,
        group_assignments_df.copy(),
        prune_ratio=float(prune_ratio),
        min_keep_per_group=int(min_keep_per_group),
    )
    decisions_df = baseline_plan['merged_scores_df'].copy()
    pruned_keys = set(baseline_plan['pruned_samples']['sample_idx'].astype(int).tolist())
    decisions_df['prune_mode'] = 'fixed'
    decisions_df['prune_selected'] = decisions_df['sample_idx'].astype(int).isin(pruned_keys)
    ranked = decisions_df[['sample_idx', 'group_id', 'image_score', 'prune_selected']].sort_values(
        ['group_id', 'image_score', 'prune_selected', 'sample_idx'],
        ascending=[True, False, False, True],
    )
    ranked['prune_rank'] = ranked.groupby('group_id').cumcount() + 1
    rank_by_sample = ranked.set_index('sample_idx')['prune_rank']
    decisions_df['prune_rank'] = decisions_df['sample_idx'].map(rank_by_sample).astype(int)
    decisions_df['prune_decision'] = np.where(decisions_df['prune_selected'], 'removed', 'retained')
    decisions_df['prune_group_reason'] = 'fixed_ratio'
    decisions_df['prune_group_pi_high'] = np.nan
    decisions_df['prune_group_selected_active'] = np.nan
    decisions_df['prune_group_stable_active'] = np.nan
    decisions_df['prune_group_active_ratio'] = np.nan
    decisions_df['prune_group_valid_observations'] = 0
    decisions_df['prune_group_source_iters'] = ''
    decisions_df['prune_estimated_high_ratio'] = np.nan
    decisions_df['prune_requested_ratio'] = float(prune_ratio)
    group_sizes = decisions_df.groupby('group_id')['sample_idx'].transform('size').astype(int)
    removed_by_group = decisions_df.groupby('group_id')['prune_selected'].transform('sum').astype(int)
    decisions_df['prune_effective_ratio'] = removed_by_group / group_sizes
    decisions_df['prune_max_ratio'] = np.nan

    group_rows = []
    for group_id, group_df in decisions_df.groupby('group_id', sort=True):
        group_size = int(len(group_df))
        removed = int(group_df['prune_selected'].sum())
        group_rows.append({
            'group_id': int(group_id),
            'group_size': group_size,
            'configured_fixed_ratio': float(prune_ratio),
            'requested_remove': int(math.floor(group_size * float(prune_ratio))),
            'max_removable': int(max(0, group_size - int(min_keep_per_group))),
            'removed': removed,
            'retained': int(group_size - removed),
            'effective_prune_ratio': float(removed / group_size) if group_size > 0 else 0.0,
            'decision_reason': 'fixed_ratio',
        })
    group_summary_df = pd.DataFrame(group_rows)
    pruned_samples = decisions_df.loc[decisions_df['prune_selected']].copy().reset_index(drop=True)
    kept_samples = decisions_df.loc[~decisions_df['prune_selected']].copy().reset_index(drop=True)
    summary = dict(baseline_plan['summary'])
    summary.update({
        'prune_mode': 'fixed',
        'gbps_prune_ratio': float(prune_ratio),
        'effective_prune_ratio': float(len(pruned_samples) / len(decisions_df)) if len(decisions_df) > 0 else 0.0,
    })
    return {
        'mode': 'remove',
        'merged_scores_df': decisions_df,
        'prune_decisions_df': decisions_df,
        'group_summary_df': group_summary_df,
        'pruned_samples': pruned_samples,
        'kept_samples': kept_samples,
        'retained_index_map': _build_retained_index_map(kept_samples),
        'summary': summary,
    }


def build_tailguard_head_delete_plan(selected_scored_df: pd.DataFrame,
                                       head_group_assignments_df: pd.DataFrame,
                                       metadata_df: pd.DataFrame,
                                       args,
                                       gbps_dir: Optional[str] = None,
                                       selected_iter: Optional[int] = None,
                                       force_no_delete_reason: Optional[str] = None):
    selected_scores = enrich_scores_with_tailguard_metadata(selected_scored_df, metadata_df)
    h_scores_df = selected_scores.loc[selected_scores['tail_candidate'] == 0].copy().reset_index(drop=True)
    prune_mode = str(getattr(
        args,
        'gbps_prune_mode',
        TAILGUARD_FINAL_DEFAULTS['gbps_prune_mode'],
    )).strip().lower()
    min_keep_per_group = int(getattr(
        args,
        'gbps_min_keep_per_group',
        TAILGUARD_FINAL_DEFAULTS['gbps_min_keep_per_group'],
    ))
    force_no_delete = force_no_delete_reason is not None

    if prune_mode == 'fixed':
        delete_plan = _build_fixed_head_delete_plan(
            h_scores_df,
            head_group_assignments_df,
            prune_ratio=(
                0.0
                if force_no_delete
                else float(getattr(args, 'gbps_prune_ratio', TAILGUARD_FINAL_DEFAULTS['gbps_prune_ratio']))
            ),
            min_keep_per_group=min_keep_per_group,
        )
    elif prune_mode == 'adaptive':
        if gbps_dir is None or selected_iter is None:
            raise ValueError('adaptive H pruning requires gbps_dir and selected_iter')
        group_context_df = load_tailguard_head_prune_group_context(gbps_dir, int(selected_iter), args)
        delete_plan = _build_adaptive_capped_head_delete_plan(
            h_scores_df,
            head_group_assignments_df,
            group_context_df,
            max_prune_ratio=(
                0.0
                if force_no_delete
                else float(getattr(
                    args,
                    'gbps_prune_max_ratio',
                    TAILGUARD_FINAL_DEFAULTS['gbps_prune_max_ratio'],
                ))
            ),
            min_keep_per_group=min_keep_per_group,
        )
    else:
        raise ValueError('unsupported gbps_prune_mode: {}'.format(prune_mode))

    h_clean_df = delete_plan['kept_samples'].copy().reset_index(drop=True)
    h_removed_df = delete_plan['pruned_samples'].copy().reset_index(drop=True)
    summary = dict(delete_plan['summary'])
    summary.update({
        'gbps_prune_mode': prune_mode,
        'gbps_prune_max_ratio': float(getattr(
            args,
            'gbps_prune_max_ratio',
            TAILGUARD_FINAL_DEFAULTS['gbps_prune_max_ratio'],
        )),
        'gbps_prune_ratio': float(getattr(
            args,
            'gbps_prune_ratio',
            TAILGUARD_FINAL_DEFAULTS['gbps_prune_ratio'],
        )),
        'gbps_prune_stable_window': int(getattr(
            args,
            'gbps_prune_stable_window',
            TAILGUARD_FINAL_DEFAULTS['gbps_prune_stable_window'],
        )),
        'gbps_prune_stable_min_observations': int(getattr(
            args,
            'gbps_prune_stable_min_observations',
            TAILGUARD_FINAL_DEFAULTS['gbps_prune_stable_min_observations'],
        )),
        'gbps_prune_min_active_ratio': float(getattr(
            args,
            'gbps_prune_min_active_ratio',
            TAILGUARD_FINAL_DEFAULTS['gbps_prune_min_active_ratio'],
        )),
        'tailguard_h_only_removal': True,
        'h_zero_removal_fallback': bool(force_no_delete),
        'h_zero_removal_fallback_reason': force_no_delete_reason,
        'num_h_clean': int(len(h_clean_df)),
        'num_h_removed': int(len(h_removed_df)),
        'num_t_untouched_by_h_removal': int((selected_scores['tail_candidate'] == 1).sum()),
    })
    if force_no_delete and len(h_removed_df) != 0:
        raise RuntimeError('H zero-removal fallback produced a non-empty removal set')
    return {
        'mode': 'h_only_remove',
        'prune_mode': prune_mode,
        'selected_scores_df': selected_scores,
        'h_scores_df': h_scores_df,
        'h_clean_samples': h_clean_df,
        'h_removed_samples': h_removed_df,
        'h_prune_decisions_df': delete_plan['prune_decisions_df'],
        'h_prune_group_summary_df': delete_plan['group_summary_df'],
        'retained_index_map_h_only': _build_retained_index_map(h_clean_df),
        'delete_plan': delete_plan,
        'baseline_plan': delete_plan,
        'summary': summary,
    }


def _list_window_iters(gbps_dir: str, selected_iter: int, stable_window: int):
    iter_numbers = []
    for name in os.listdir(gbps_dir):
        iter_number = _iter_number_from_name(name)
        if iter_number is not None and iter_number <= int(selected_iter):
            if os.path.isfile(os.path.join(gbps_dir, name, 'train_scores.csv')):
                iter_numbers.append(iter_number)
    iter_numbers = sorted(iter_numbers)
    return iter_numbers[-max(1, int(stable_window)):]


def _normal_pdf(x, mu, sigma):
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * ((float(x) - float(mu)) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _posterior_high_from_metrics(image_score: float, metrics_row: pd.Series):
    required = ['score_median', 'score_mad', 'mu_low', 'mu_high', 'sigma_low', 'sigma_high', 'pi_low', 'pi_high']
    if any(pd.isna(metrics_row.get(column)) for column in required):
        return None
    z_score = (float(image_score) - float(metrics_row['score_median'])) / max(float(metrics_row['score_mad']), 1e-6)
    low = float(metrics_row['pi_low']) * _normal_pdf(z_score, metrics_row['mu_low'], metrics_row['sigma_low'])
    high = float(metrics_row['pi_high']) * _normal_pdf(z_score, metrics_row['mu_high'], metrics_row['sigma_high'])
    denom = low + high
    if denom <= 0:
        return None
    return float(high / denom)


def classify_tail_attached_stable_risk(tail_attached_df: pd.DataFrame,
                                       gbps_dir: str,
                                       selected_iter: Optional[int],
                                       args):
    attached_df = tail_attached_df.copy().reset_index(drop=True)
    stable_columns = {
        'stable_p_high_median': 0.0,
        'stable_active_ratio': 0.0,
        'stable_valid_observations': 0,
        'stable_risk_source_iters': '',
        'stable_selected_group_active': False,
        'stable_head_noise': False,
    }
    if len(attached_df) == 0:
        for column, value in stable_columns.items():
            attached_df[column] = value
        return attached_df.copy(), attached_df.copy(), attached_df.copy()
    if selected_iter is None:
        for column, value in stable_columns.items():
            attached_df[column] = value
        return attached_df.copy(), attached_df.iloc[0:0].copy(), attached_df.copy()

    window_iters = _list_window_iters(gbps_dir, int(selected_iter), int(getattr(args, 'tg_stable_window', 3)))
    p_high_thr = float(getattr(
        args,
        'tg_t_attached_noise_p_high_thr',
        TAILGUARD_FINAL_DEFAULTS['tg_t_attached_noise_p_high_thr'],
    ))
    min_valid = int(getattr(args, 'tg_stable_min_observations', 1))
    window_data = {}
    for iter_number in window_iters:
        iter_dir = os.path.join(gbps_dir, 'iter_{:05d}'.format(int(iter_number)))
        score_path = os.path.join(iter_dir, 'train_scores.csv')
        metrics_path = os.path.join(iter_dir, 'h_group_metrics.csv')
        if not os.path.isfile(score_path) or not os.path.isfile(metrics_path):
            continue
        train_scores_df = pd.read_csv(score_path, usecols=['sample_idx', 'image_score'])
        group_metrics_df = pd.read_csv(metrics_path)
        window_data[int(iter_number)] = (
            train_scores_df.set_index('sample_idx')['image_score'],
            group_metrics_df.set_index('group_id', drop=False),
        )
    rows = []

    for _, row in attached_df.iterrows():
        sample_idx = int(row['sample_idx'])
        best_group_id = int(row['best_group_id'])
        p_high_values = []
        active_values = []
        source_iters = []
        selected_active = False
        for iter_number, (image_scores_by_sample, group_metrics_by_group) in window_data.items():
            if sample_idx not in image_scores_by_sample.index or best_group_id not in group_metrics_by_group.index:
                continue
            metric_row = group_metrics_by_group.loc[best_group_id]
            if isinstance(metric_row, pd.DataFrame):
                metric_row = metric_row.iloc[0]
            p_high = _posterior_high_from_metrics(float(image_scores_by_sample.loc[sample_idx]), metric_row)
            if p_high is None:
                continue
            is_active = bool(metric_row.get('is_active_group', False))
            if int(iter_number) == int(selected_iter):
                selected_active = is_active
            p_high_values.append(float(p_high))
            active_values.append(1.0 if is_active else 0.0)
            source_iters.append(int(iter_number))

        valid_observations = len(p_high_values)
        active_ratio = float(np.mean(active_values)) if len(active_values) > 0 else 0.0
        p_high_median = float(np.median(p_high_values)) if len(p_high_values) > 0 else 0.0
        is_noise = bool(
            valid_observations >= min_valid
            and selected_active
            and active_ratio > 0.5
            and p_high_median > p_high_thr
        )
        out_row = row.to_dict()
        out_row.update({
            'stable_p_high_median': p_high_median,
            'stable_active_ratio': active_ratio,
            'stable_valid_observations': int(valid_observations),
            'stable_risk_source_iters': ','.join(str(value) for value in source_iters),
            'stable_selected_group_active': bool(selected_active),
            'stable_head_noise': bool(is_noise),
        })
        rows.append(out_row)

    risk_df = pd.DataFrame(rows)
    if len(risk_df) == 0:
        for column, value in stable_columns.items():
            attached_df[column] = value
        return attached_df.copy(), attached_df.iloc[0:0].copy(), attached_df.copy()
    noise_df = risk_df.loc[risk_df['stable_head_noise']].copy().reset_index(drop=True)
    normal_df = risk_df.loc[~risk_df['stable_head_noise']].copy().reset_index(drop=True)
    return normal_df, noise_df, risk_df


def build_tailguard_stage2_plan(h_clean_df: pd.DataFrame,
                                  h_removed_df: pd.DataFrame,
                                  tail_open_df: pd.DataFrame,
                                  tail_head_normal_df: pd.DataFrame,
                                  tail_head_noise_df: pd.DataFrame,
                                  all_samples_df: Optional[pd.DataFrame] = None):
    source_frames = [
        h_clean_df,
        h_removed_df,
        tail_open_df,
        tail_head_normal_df,
        tail_head_noise_df,
    ]

    def _empty_sample_frame():
        """Keep the Stage-2 sample schema even when a partition is empty."""
        candidates = [all_samples_df] + source_frames
        for frame in candidates:
            if frame is not None and 'sample_idx' in frame.columns:
                return frame.iloc[0:0].copy()
        return pd.DataFrame(columns=['sample_idx'])

    retained_frames = [
        frame for frame in [h_clean_df, tail_head_normal_df, tail_open_df]
        if frame is not None and len(frame) > 0
    ]
    removed_frames = [
        frame for frame in [h_removed_df, tail_head_noise_df]
        if frame is not None and len(frame) > 0
    ]
    retained_df = (
        pd.concat(retained_frames, ignore_index=True, sort=False)
        if len(retained_frames) > 0 else _empty_sample_frame()
    )
    removed_df = (
        pd.concat(removed_frames, ignore_index=True, sort=False)
        if len(removed_frames) > 0 else _empty_sample_frame()
    )
    if len(retained_df) > 0:
        retained_df = retained_df.drop_duplicates(subset=['sample_idx']).reset_index(drop=True)
    if len(removed_df) > 0:
        removed_df = removed_df.drop_duplicates(subset=['sample_idx']).reset_index(drop=True)
    summary = {
        'num_stage2_retained': int(len(retained_df)),
        'num_stage2_removed': int(len(removed_df)),
        'num_h_clean': int(len(h_clean_df)),
        'num_h_removed': int(len(h_removed_df)),
        'num_tail_open': int(len(tail_open_df)),
        'num_tail_head_normal': int(len(tail_head_normal_df)),
        'num_tail_head_noise': int(len(tail_head_noise_df)),
    }
    if all_samples_df is None:
        all_frames = [frame for frame in [retained_df, removed_df] if frame is not None and len(frame) > 0]
        all_samples_df = pd.concat(all_frames, ignore_index=True, sort=False) if len(all_frames) > 0 else None
    return {
        'retained_samples': retained_df,
        'removed_samples': removed_df,
        'retained_index_map': _build_retained_index_map(retained_df, all_samples_df=all_samples_df),
        'summary': summary,
    }
