"""TailSampler, embeddings, and H-only grouping preparation for TailGuard."""

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tailguard.method.grouping import extract_image_embeddings, fit_latent_groups
from tailguard.method.tail_sampling import _compute_ths, build_tail_sampler, compute_self_sim
from tailguard.reporting.tail_sampling import run_tail_sampler_analysis, save_tail_sampler_artifacts
from tailguard.reporting.artifacts import save_tailguard_prepare_artifacts
from tailguard.config import TAILGUARD_FINAL_DEFAULTS


DEFAULT_K_CANDIDATES = tuple(
    int(value) for value in TAILGUARD_FINAL_DEFAULTS['gbps_k_candidates'].split(',')
)

TAILSAMPLER_CANDIDATE_COLUMNS = [
    'sample_idx',
    'sample_key',
    'img_path',
    'tail_candidate',
    'pred_class_size',
    'tail_score',
    'adaptive_angle',
    'adaptive_angle_source',
]

HEAD_GROUP_ASSIGNMENT_COLUMNS = [
    'sample_key',
    'sample_idx',
    'img_path',
    'base_idx',
    'group_id',
    'group_size',
]

METHOD_METADATA_COLUMNS = [
    'sample_idx',
    'sample_key',
    'img_path',
    'base_idx',
    'tail_candidate',
    'pred_class_size',
    'tail_score',
    'adaptive_angle',
    'group_id',
    'group_size',
]

ANALYSIS_METADATA_COLUMNS = [
    'sample_idx',
    'sample_key',
    'img_path',
    'base_idx',
    'class_id',
    'class_name',
    'is_contaminated',
]


def _analysis_metadata_view(full_metadata_df: pd.DataFrame) -> pd.DataFrame:
    available_columns = [column for column in ANALYSIS_METADATA_COLUMNS if column in full_metadata_df.columns]
    analysis_metadata_df = full_metadata_df[available_columns].copy()
    for column in ANALYSIS_METADATA_COLUMNS:
        if column not in analysis_metadata_df.columns:
            analysis_metadata_df[column] = np.nan
    return analysis_metadata_df[ANALYSIS_METADATA_COLUMNS]


def _parse_k_candidates(values):
    if values is None:
        return DEFAULT_K_CANDIDATES
    if isinstance(values, str):
        return tuple(int(part.strip()) for part in values.split(',') if part.strip())
    return tuple(int(value) for value in values)


def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor.float(), dim=-1)


def _as_float_tensor(embeddings: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(embeddings, dtype=np.float32))


def _compute_adaptive_angles(features: torch.Tensor, args):
    default_angle = float(getattr(args, 'tg_default_adaptive_angle', 5.0))
    sampler_type = getattr(args, 'tailsampler_type', TAILGUARD_FINAL_DEFAULTS['tailsampler_type'])
    threshold_type = getattr(args, 'tailsampler_th_type', None)

    if sampler_type == 'adaptive':
        threshold_type = 'double_max_step' if threshold_type is None else threshold_type
    elif sampler_type == 'adaptive_trim_mode':
        threshold_type = 'trim_min' if threshold_type is None else threshold_type
    else:
        angles = torch.full((features.shape[0],), default_angle, dtype=torch.float32)
        return angles, 'default_non_adaptive'

    try:
        self_sim = compute_self_sim(features.detach().cpu())
        thresholds = _compute_ths(self_sim, threshold_type).reshape(-1)
        thresholds = torch.clamp(thresholds.float(), min=-1.0, max=1.0)
        angles = torch.rad2deg(torch.acos(thresholds))
        angles = torch.nan_to_num(angles, nan=default_angle, posinf=default_angle, neginf=default_angle)
        angles = torch.clamp(angles, min=default_angle)
        return angles.cpu(), 'adaptive_threshold_{}'.format(threshold_type)
    except Exception:
        angles = torch.full((features.shape[0],), default_angle, dtype=torch.float32)
        return angles, 'default_after_angle_error'


def _extract_tail_candidates(embeddings: np.ndarray, metadata_df: pd.DataFrame, args):
    sampler, sampler_name = build_tail_sampler(
        sampler_type=getattr(args, 'tailsampler_type', TAILGUARD_FINAL_DEFAULTS['tailsampler_type']),
        threshold_type=getattr(args, 'tailsampler_th_type', None),
        vote_type=getattr(args, 'tailsampler_vote_type', None),
        percentile=float(getattr(args, 'tailsampler_percentile', 0.15)),
        plateau_gap_guard=bool(getattr(
            args,
            'tailsampler_plateau_gap_guard',
            TAILGUARD_FINAL_DEFAULTS['tailsampler_plateau_gap_guard'],
        )),
    )
    features = _as_float_tensor(embeddings)
    _, tail_indices, pred_class_sizes = sampler.run(features, return_class_sizes=True)
    tail_indices = tail_indices.detach().cpu().numpy().astype(int)

    tail_mask = np.zeros(len(metadata_df), dtype=np.int64)
    tail_mask[tail_indices] = 1
    pred_class_sizes = pred_class_sizes.detach().cpu().numpy().astype(np.float32)
    tail_scores = 1.0 / np.maximum(pred_class_sizes, 1.0)
    adaptive_angles, angle_source = _compute_adaptive_angles(features, args)

    candidates_df = metadata_df[['sample_idx', 'sample_key', 'img_path']].copy()
    candidates_df['tail_candidate'] = tail_mask.astype(int)
    candidates_df['pred_class_size'] = pred_class_sizes.astype(float)
    candidates_df['tail_score'] = tail_scores.astype(float)
    candidates_df['adaptive_angle'] = adaptive_angles.numpy().astype(float)
    candidates_df['adaptive_angle_source'] = angle_source
    partition_diagnostics = getattr(sampler, 'last_partition_diagnostics', None)
    if partition_diagnostics is None:
        partition_diagnostics = {
            'guard_name': None,
            'enabled': False,
            'triggered': False,
            'status': 'unavailable_for_sampler',
            'num_samples': int(len(candidates_df)),
            'num_selected_raw': int(tail_mask.sum()),
            'num_selected_final': int(tail_mask.sum()),
            'num_removed_by_guard': 0,
            'selected_ratio_raw': float(tail_mask.mean()) if len(tail_mask) > 0 else 0.0,
            'selected_ratio_final': float(tail_mask.mean()) if len(tail_mask) > 0 else 0.0,
            'original_cutoff': None,
            'effective_cutoff': None,
            'num_inferred_classes': 0,
            'boundary_multiplicity': 0,
            'gap_lower': None,
            'gap_upper': None,
            'dominant_log_gap': None,
        }
    return (
        candidates_df[TAILSAMPLER_CANDIDATE_COLUMNS].copy(),
        sampler_name,
        angle_source,
        dict(partition_diagnostics),
    )


def _build_head_group_assignments(group_embeddings: np.ndarray, metadata_df: pd.DataFrame, candidates_df: pd.DataFrame, args):
    head_mask = candidates_df['tail_candidate'].to_numpy(dtype=int) == 0
    if int(head_mask.sum()) == 0:
        raise ValueError('TailGuard prepare requires at least one non-tail H sample')

    head_metadata_df = metadata_df.loc[head_mask].copy().reset_index(drop=True)
    head_embeddings = np.asarray(group_embeddings, dtype=np.float32)[head_mask]
    group_assignments, group_info = fit_latent_groups(
        head_embeddings,
        head_metadata_df['sample_key'].tolist(),
        method=getattr(args, 'gbps_grouping_method', 'kmeans'),
        num_groups=getattr(args, 'gbps_num_groups', None),
        auto_k=bool(getattr(args, 'gbps_auto_k', TAILGUARD_FINAL_DEFAULTS['gbps_auto_k'])),
        k_candidates=_parse_k_candidates(getattr(args, 'gbps_k_candidates', DEFAULT_K_CANDIDATES)),
        pca_dim=getattr(args, 'gbps_pca_dim', 64),
        random_state=0,
    )

    group_df = head_metadata_df[['sample_key', 'sample_idx', 'img_path', 'base_idx']].copy()
    group_df['group_id'] = group_df['sample_key'].map({int(key): int(value) for key, value in group_assignments.items()})
    group_sizes = group_df.groupby('group_id').size().rename('group_size').reset_index()
    group_df = group_df.merge(group_sizes, on='group_id', how='left')
    group_df['group_id'] = group_df['group_id'].astype(int)
    group_df['group_size'] = group_df['group_size'].astype(int)
    group_info.update({
        'tailguard_h_only': True,
        'num_head_samples': int(len(group_df)),
        'num_tail_samples_excluded': int((~head_mask).sum()),
    })
    return group_df[HEAD_GROUP_ASSIGNMENT_COLUMNS].copy(), group_info


@torch.no_grad()
def prepare_tailguard_metadata(model,
                                 train_eval_dataloader,
                                 device,
                                 output_dir: Optional[str],
                                 args,
                                 manifest_path: Optional[str] = None,
                                 manifest_requested_path: Optional[str] = None):
    tail_embedding_source = getattr(args, 'tg_tail_embedding_source', None) or getattr(args, 'tailsampler_embedding_source', 'encoder_cls')
    group_embedding_source = getattr(args, 'tg_group_embedding_source', None) or getattr(args, 'gbps_embedding_source', 'encoder')

    tail_embeddings, metadata_df = extract_image_embeddings(
        model,
        train_eval_dataloader,
        device,
        embedding_source=tail_embedding_source,
    )
    metadata_df = metadata_df.copy()
    for column in ['sample_idx', 'sample_key', 'base_idx']:
        metadata_df[column] = metadata_df[column].astype(int)
    tail_order = np.argsort(metadata_df['sample_idx'].to_numpy())
    tail_embeddings = tail_embeddings[tail_order]
    metadata_df = metadata_df.iloc[tail_order].reset_index(drop=True)

    if len(metadata_df) == 0:
        raise ValueError('TailGuard prepare requires non-empty train metadata')

    if group_embedding_source == tail_embedding_source:
        group_embeddings = tail_embeddings
        group_metadata_df = metadata_df.copy()
    else:
        group_embeddings, group_metadata_df = extract_image_embeddings(
            model,
            train_eval_dataloader,
            device,
            embedding_source=group_embedding_source,
        )
        group_metadata_df = group_metadata_df.copy()
        group_order = np.argsort(group_metadata_df['sample_idx'].to_numpy())
        group_embeddings = group_embeddings[group_order]
        group_metadata_df = group_metadata_df.iloc[group_order].reset_index(drop=True)
        if group_metadata_df['sample_idx'].astype(int).tolist() != metadata_df['sample_idx'].astype(int).tolist():
            raise ValueError('group embedding metadata mismatch during TailGuard prepare')

    candidates_df, sampler_name, angle_source, partition_diagnostics = _extract_tail_candidates(
        tail_embeddings,
        metadata_df,
        args,
    )
    head_group_assignments_df, group_info = _build_head_group_assignments(group_embeddings, metadata_df, candidates_df, args)

    full_metadata_df = metadata_df.merge(
        candidates_df,
        on=['sample_idx', 'sample_key', 'img_path'],
        how='left',
        validate='one_to_one',
    )
    full_metadata_df = full_metadata_df.merge(
        head_group_assignments_df,
        on=['sample_idx', 'sample_key', 'img_path', 'base_idx'],
        how='left',
        validate='one_to_one',
    )
    full_metadata_df['tail_candidate'] = full_metadata_df['tail_candidate'].astype(int)
    full_metadata_df['pred_class_size'] = full_metadata_df['pred_class_size'].astype(float)
    full_metadata_df['tail_score'] = full_metadata_df['tail_score'].astype(float)
    full_metadata_df['adaptive_angle'] = full_metadata_df['adaptive_angle'].astype(float)
    full_metadata_df['group_id'] = full_metadata_df['group_id'].fillna(-1).astype(int)
    full_metadata_df['group_size'] = full_metadata_df['group_size'].fillna(0).astype(int)
    full_metadata_df = full_metadata_df.sort_values('sample_idx').reset_index(drop=True)

    train_metadata_df = full_metadata_df[METHOD_METADATA_COLUMNS].copy()
    analysis_metadata_df = _analysis_metadata_view(full_metadata_df)
    candidates_df = candidates_df.sort_values('sample_idx').reset_index(drop=True)
    head_group_assignments_df = head_group_assignments_df.sort_values('sample_idx').reset_index(drop=True)

    cls_payload = None
    if bool(getattr(args, 'tg_save_cls_embeddings', True)):
        cls_embeddings, cls_metadata_df = extract_image_embeddings(
            model,
            train_eval_dataloader,
            device,
            embedding_source='encoder_cls',
        )
        cls_metadata_df = cls_metadata_df.copy()
        cls_order = np.argsort(cls_metadata_df['sample_idx'].to_numpy())
        cls_embeddings = cls_embeddings[cls_order]
        cls_metadata_df = cls_metadata_df.iloc[cls_order].reset_index(drop=True)
        if cls_metadata_df['sample_idx'].astype(int).tolist() != train_metadata_df['sample_idx'].astype(int).tolist():
            raise ValueError('cls embedding metadata mismatch during TailGuard prepare')

        cls_payload = {
            'sample_idx': cls_metadata_df['sample_idx'].astype(int).tolist(),
            'img_path': cls_metadata_df['img_path'].astype(str).tolist(),
            'embeddings': _normalize_rows(_as_float_tensor(cls_embeddings)).cpu(),
            'embedding_source': 'encoder_cls',
        }
    grouping_payload = {
        'sample_idx': metadata_df['sample_idx'].astype(int).tolist(),
        'img_path': metadata_df['img_path'].astype(str).tolist(),
        'embeddings': _normalize_rows(_as_float_tensor(group_embeddings)).cpu(),
        'embedding_source': group_embedding_source,
    }

    analysis_summary = None
    analysis_saved = None
    if bool(getattr(args, 'tg_save_tailsampler_analysis', True)):
        try:
            if getattr(args, 'tailsampler_dataset_name', None) is None:
                setattr(args, 'tailsampler_dataset_name', os.path.basename(os.path.normpath(str(getattr(args, 'data_path', '')))))
            analysis_summary, analysis_details_df = run_tail_sampler_analysis(
                tail_embeddings,
                full_metadata_df,
                args,
                plateau_gap_guard=bool(getattr(
                    args,
                    'tailsampler_plateau_gap_guard',
                    TAILGUARD_FINAL_DEFAULTS['tailsampler_plateau_gap_guard'],
                )),
            )
            if output_dir is not None:
                analysis_saved = save_tail_sampler_artifacts(
                    os.path.join(output_dir, 'tail_sampler_analysis_only'),
                    analysis_summary,
                analysis_details_df,
                metadata={
                    'analysis_only': True,
                    'not_used_by_tailguard_decision_logic': True,
                    'partition_matches_tailguard_decision_logic': True,
                        'tail_embedding_source': tail_embedding_source,
                        'gt_mode': getattr(args, 'tailsampler_gt_mode', 'dataset_rule'),
                        'dataset_name': getattr(args, 'tailsampler_dataset_name', None),
                    },
                    save_details=bool(getattr(args, 'tg_save_tailsampler_analysis_details', True)),
                )
        except Exception as exc:
            analysis_summary = {
                'status': 'analysis_failed',
                'error': str(exc),
                'analysis_only': True,
                'not_used_by_tailguard_decision_logic': True,
            }

    prepare_summary = {
        'num_samples': int(len(train_metadata_df)),
        'num_tail_candidates': int(train_metadata_df['tail_candidate'].sum()),
        'num_head_candidates': int((train_metadata_df['tail_candidate'] == 0).sum()),
        'tail_candidate_ratio': float(train_metadata_df['tail_candidate'].mean()),
        'num_head_groups': int(head_group_assignments_df['group_id'].nunique()),
        'tail_sampler_type': getattr(args, 'tailsampler_type', TAILGUARD_FINAL_DEFAULTS['tailsampler_type']),
        'tail_sampler_impl': sampler_name,
        'tail_embedding_source': tail_embedding_source,
        'group_embedding_source': group_embedding_source,
        'adaptive_angle_source': angle_source,
        'tail_partition_guard': partition_diagnostics,
        'analysis_metadata_provenance': {
            'manifest_path': manifest_path,
            'manifest_requested_path': manifest_requested_path,
            'label_column': 'is_contaminated',
        },
        'group_info': group_info,
        'tail_sampler_analysis_only_summary': analysis_summary,
        'tail_sampler_analysis_only_artifacts': analysis_saved,
    }

    saved = None
    if output_dir is not None:
        metadata = {
            'tail_sampler_type': getattr(args, 'tailsampler_type', TAILGUARD_FINAL_DEFAULTS['tailsampler_type']),
            'tailsampler_th_type': getattr(args, 'tailsampler_th_type', None),
            'tailsampler_vote_type': getattr(args, 'tailsampler_vote_type', None),
            'tailsampler_percentile': float(getattr(args, 'tailsampler_percentile', 0.15)),
            'tailsampler_plateau_gap_guard': bool(getattr(
                args,
                'tailsampler_plateau_gap_guard',
                TAILGUARD_FINAL_DEFAULTS['tailsampler_plateau_gap_guard'],
            )),
            'tail_partition_guard': partition_diagnostics,
            'tail_embedding_source': tail_embedding_source,
            'group_embedding_source': group_embedding_source,
            'grouping_method': getattr(args, 'gbps_grouping_method', 'kmeans'),
            'gbps_num_groups': getattr(args, 'gbps_num_groups', None),
            'gbps_auto_k': bool(getattr(args, 'gbps_auto_k', TAILGUARD_FINAL_DEFAULTS['gbps_auto_k'])),
            'gbps_k_candidates': list(_parse_k_candidates(getattr(args, 'gbps_k_candidates', DEFAULT_K_CANDIDATES))),
            'gbps_pca_dim': getattr(args, 'gbps_pca_dim', 64),
            'summary': prepare_summary,
        }
        saved = save_tailguard_prepare_artifacts(
            output_dir,
            candidates_df,
            train_metadata_df,
            head_group_assignments_df,
            metadata=metadata,
            cls_embeddings_payload=cls_payload,
            grouping_embeddings_payload=grouping_payload if bool(getattr(args, 'tg_save_grouping_embeddings', True)) else None,
            analysis_metadata_df=analysis_metadata_df,
        )

    return {
        'candidates_df': candidates_df,
        'head_group_assignments_df': head_group_assignments_df,
        'train_metadata_df': train_metadata_df,
        'analysis_metadata_df': analysis_metadata_df,
        'full_metadata_df': full_metadata_df,
        'cls_embeddings_payload': cls_payload,
        'grouping_embeddings_payload': grouping_payload,
        'group_info': group_info,
        'summary': prepare_summary,
        'saved': saved,
    }
