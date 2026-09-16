"""Canonical TailGuard runner for MVTec-compatible datasets."""

import argparse
import json
import os
import random
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tailguard.engine.trainer import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CROP_SIZE,
    DEFAULT_ENCODER_NAME,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_TOTAL_ITERS,
    build_datasets,
    build_model,
    build_optimizer_and_scheduler,
    build_train_dataloader,
    build_train_eval_dataloader,
    _checkpoint_args_get,
    evaluate_model,
    get_logger,
    load_model_from_train_checkpoint,
    rebuild_pruned_train_loaders,
    save_train_checkpoint,
    score_trainset,
    setup_seed,
    should_run_check,
    train_step,
)
from tailguard.method.dhp_utils import evaluate_gbps_ci_peak_trigger, resolve_gbps_best_scored_df
from tailguard.reporting.artifacts import (
    save_tailguard_attachment_replay_config,
    save_tailguard_attachment_artifacts,
    save_tailguard_denoising_diagnostics,
    save_tailguard_head_prune_artifacts,
    save_tailguard_pseudoclass_artifacts,
    save_tailguard_pseudoclass_report_artifacts,
    save_tailguard_gbps_iteration_artifacts,
    save_tailguard_memory_artifacts,
    save_tailguard_stage2_artifacts,
    save_tailguard_summary,
    save_tailguard_trigger_summary,
)
from tailguard.method.trp import build_tail_attachment_plan
from tailguard.method.dhp import (
    build_tailguard_head_delete_plan,
    build_tailguard_stage2_plan,
    load_tailguard_selected_checkpoint,
    run_tailguard_gbps_iteration,
    save_tailguard_selected_checkpoint,
)
from tailguard.reporting.diagnostics import (
    build_tailguard_denoising_diagnostics,
    build_tailguard_pseudoclass_report,
)
from tailguard.method.cte import (
    build_tailguard_pseudoclass_memory_system,
    run_pseudoclass_memory_evaluation,
)
from tailguard.data.profiles import (
    MVTec_PROFILE,
    dataset_profile_names,
    get_dataset_profile,
    profile_provenance,
    validate_profile_contract,
)
from tailguard.method.preparation import prepare_tailguard_metadata
from tailguard.method.registry import build_tailguard_pseudoclass_registry
from tailguard.config import (
    TAILGUARD_CONFIG_PROFILE,
    TAILGUARD_CONFIG_SCHEMA_VERSION,
    TAILGUARD_FINAL_DEFAULTS,
    TAILGUARD_METHOD_MODES,
    TAILGUARD_METHOD_NAME,
    TAILGUARD_PRUNE_MODES,
)
from tailguard.data.manifests import load_injected_manifest

warnings = __import__('warnings')
warnings.filterwarnings('ignore')


MVTec_ITEM_LIST = list(MVTec_PROFILE.item_list)


def _build_retained_index_map(samples_df: pd.DataFrame, num_classes=None):
    retained_index_map = {}
    if num_classes is not None:
        for class_id in range(int(num_classes)):
            retained_index_map[int(class_id)] = []
    for class_id, class_df in samples_df.groupby('class_id', sort=True):
        ordered = class_df.sort_values(['base_idx', 'sample_idx'], kind='mergesort')
        retained_index_map[int(class_id)] = [int(value) for value in ordered['base_idx'].astype(int).tolist()]
    return retained_index_map


@contextmanager
def _preserve_rng_state():
    """Keep analysis-only loader passes from changing the training trajectory."""
    state = {
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }
    try:
        yield
    finally:
        torch.set_rng_state(state['torch'].detach().cpu().byte())
        if torch.cuda.is_available() and state['cuda'] is not None:
            torch.cuda.set_rng_state_all([
                value.detach().cpu().byte() for value in state['cuda']
            ])
        np.random.set_state(state['numpy'])
        random.setstate(state['python'])


def _mode_uses_reconciliation(method_mode):
    return method_mode in {'full_no_e', 'full'}


def _mode_uses_enhancement(method_mode):
    return method_mode in {'h_raw_e', 'full'}


def _resolve_tailguard_trigger_selection(trigger_result):
    """Choose Stage-2 score evidence without forcing an unsupported rollback."""
    status = str(trigger_result['status'])
    selected_iter = trigger_result.get('selected_iter')
    zero_removal_fallback = status == 'no_noise_forced'
    if zero_removal_fallback:
        selected_iter = trigger_result.get('summary', {}).get('iteration')
    if selected_iter is None:
        raise RuntimeError(
            'TailGuard trigger {} did not provide a score iteration for Stage 2'.format(status)
        )
    return int(selected_iter), bool(zero_removal_fallback)


def resolved_tailguard_method_config(parsed_args):
    """Return the complete method configuration resolved for one run."""
    config = {
        name: getattr(parsed_args, name, default)
        for name, default in TAILGUARD_FINAL_DEFAULTS.items()
    }
    config.update({
        'reconciliation_enabled': _mode_uses_reconciliation(parsed_args.tg_method_mode),
        'enhancement_enabled': _mode_uses_enhancement(parsed_args.tg_method_mode),
    })
    return config


def _empty_tail_affiliated_frame(tail_df):
    empty = tail_df.iloc[0:0].copy()
    if 'best_group_id' not in empty.columns:
        empty['best_group_id'] = pd.Series(dtype=int)
    return empty


def _build_h_only_stage2_plan(head_delete_plan, tail_df, all_samples_df, num_classes):
    h_clean_df = head_delete_plan['h_clean_samples'].copy()
    h_removed_df = head_delete_plan['h_removed_samples'].copy()
    retained_df = pd.concat([h_clean_df, tail_df], ignore_index=True, sort=False)
    retained_df = retained_df.drop_duplicates(subset=['sample_idx']).sort_values(
        'sample_idx', kind='mergesort'
    ).reset_index(drop=True)
    removed_df = h_removed_df.drop_duplicates(subset=['sample_idx']).sort_values(
        'sample_idx', kind='mergesort'
    ).reset_index(drop=True)
    all_ids = set(all_samples_df['sample_idx'].astype(int))
    retained_ids = set(retained_df['sample_idx'].astype(int))
    removed_ids = set(removed_df['sample_idx'].astype(int))
    tail_ids = set(tail_df['sample_idx'].astype(int))
    if retained_ids & removed_ids or retained_ids | removed_ids != all_ids:
        raise ValueError('H-only Stage 2 partitions must be disjoint and cover all samples')
    if not tail_ids.issubset(retained_ids):
        raise ValueError('H-only must retain every TailSampler candidate')
    summary = {
        'method_mode': 'h_only',
        'num_stage2_retained': int(len(retained_df)),
        'num_stage2_removed': int(len(removed_df)),
        'num_h_clean': int(len(h_clean_df)),
        'num_h_removed': int(len(h_removed_df)),
        'num_tail_candidates_retained': int(len(tail_df)),
        'num_tail_candidates_removed': 0,
        'reconciliation_performed': False,
        'enhancement_performed': False,
    }
    return {
        'retained_samples': retained_df,
        'removed_samples': removed_df,
        'retained_index_map': _build_retained_index_map(retained_df, num_classes=num_classes),
        'summary': summary,
    }


def _validate_stage2_partition(stage2_plan, all_samples_df, tail_df, method_mode):
    retained_ids = set(stage2_plan['retained_samples']['sample_idx'].astype(int))
    removed_ids = set(stage2_plan['removed_samples']['sample_idx'].astype(int))
    all_ids = set(all_samples_df['sample_idx'].astype(int))
    tail_ids = set(tail_df['sample_idx'].astype(int))
    if retained_ids & removed_ids:
        raise ValueError('{} retained/removed Stage 2 partitions overlap'.format(method_mode))
    if retained_ids | removed_ids != all_ids:
        raise ValueError('{} Stage 2 partitions do not cover the input set'.format(method_mode))
    if not tail_ids.issubset(retained_ids):
        raise ValueError('{} must retain every TailSampler candidate'.format(method_mode))


def _build_raw_tail_roles(tail_df):
    tail_open_df = tail_df.copy().sort_values('sample_idx', kind='mergesort').reset_index(drop=True)
    tail_attached_df = _empty_tail_affiliated_frame(tail_df)
    return tail_open_df, tail_attached_df, tail_attached_df.copy()


def _validate_checkpoint_profile(profile, checkpoint_metadata):
    checkpoint_args = checkpoint_metadata.get('args') or {}
    validate_profile_contract(
        profile,
        _checkpoint_args_get(checkpoint_args, 'dataset_profile', None),
        _checkpoint_args_get(checkpoint_args, 'dataset_item_list', None),
        'checkpoint',
    )


def _evaluate_current_model(model, test_data_list, item_list, device, args, print_fn=None):
    with _preserve_rng_state():
        return evaluate_model(
            model,
            test_data_list,
            item_list,
            device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            max_ratio=float(args.max_ratio),
            resize_mask=int(args.resize_mask),
            print_fn=print_fn,
        )


def _print_eval_summary(summary, print_fn):
    if summary is None:
        return
    print_fn('I-AUROC      {:.2f}'.format(summary['I-AUROC']))
    print_fn('I-AP         {:.2f}'.format(summary['I-AP']))
    print_fn('I-F1         {:.2f}'.format(summary['I-F1']))
    print_fn('P-AUROC      {:.2f}'.format(summary['P-AUROC']))
    print_fn('P-AP         {:.2f}'.format(summary['P-AP']))
    print_fn('P-F1         {:.2f}'.format(summary['P-F1']))
    print_fn('P-AUPRO      {:.2f}'.format(summary['P-AUPRO']))


def train(item_list):
    setup_seed(1)
    method_mode = str(args.tg_method_mode)
    total_iters = int(args.total_iters)
    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)
    train_start_time = time.time()

    final_eval_summary = None
    memory_eval_summary = None
    metrics_by_mode = None
    memory_saved = None
    stage2_plan = None
    attachment_saved = None
    head_prune_saved = None
    stage2_saved = None
    denoising_diagnostics_artifacts = None
    denoising_diagnostics_summary = None
    pseudoclass_registry = None
    pseudoclass_artifacts = None
    pseudoclass_report_artifacts = None
    pseudoclass_status = 'not_performed'
    pseudoclass_reason = None
    gbps_trigger_summary = None
    selected_checkpoint_path = os.path.join(args.tg_checkpoint_dir, 'tailguard_selected_checkpoint.pt')

    if args.checkpoint_path:
        model, trainable, checkpoint_metadata = load_model_from_train_checkpoint(args.checkpoint_path, device)
        _validate_checkpoint_profile(get_dataset_profile(args.dataset_profile), checkpoint_metadata)
        image_size = int(checkpoint_metadata['image_size'])
        crop_size = int(checkpoint_metadata['crop_size'])
        print_fn('loaded checkpoint: {}'.format(args.checkpoint_path))
    else:
        model, trainable = build_model(device, encoder_name=args.encoder_name)
        checkpoint_metadata = None
        image_size = int(args.image_size)
        crop_size = int(args.crop_size)
        print_fn('built model from encoder: {}'.format(args.encoder_name))

    optimizer, lr_scheduler = build_optimizer_and_scheduler(
        trainable,
        total_iters=total_iters,
        lr=float(args.lr),
        final_lr=float(args.final_lr),
        warmup_iters=int(args.warmup_iters),
        weight_decay=float(args.weight_decay),
    )

    base_train_data_list, test_data_list = build_datasets(
        args.data_path,
        item_list,
        image_size=image_size,
        crop_size=crop_size,
    )
    contaminated_paths, manifest_path = load_injected_manifest(
        args.data_path,
        Path(__file__).resolve().parents[2],
        manifest_path=args.diag_manifest_path,
    )
    if contaminated_paths is None:
        print_fn('tailguard diagnosis: contamination manifest not found, contamination-aware logging will be unavailable.')
    else:
        print_fn('tailguard diagnosis: loaded contamination manifest {}.'.format(manifest_path))

    full_train_data, full_train_dataloader = build_train_dataloader(
        base_train_data_list,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    # Metadata preparation is outside the reconstruction optimization path.
    # Rewind to the post-model seed before the first training batch in every mode.
    post_model_rng_state = {
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }
    full_train_eval_dataloader = build_train_eval_dataloader(
        base_train_data_list,
        data_root=args.data_path,
        batch_size=args.diag_batch_size,
        num_workers=args.diag_num_workers,
        item_list=item_list,
        contaminated_paths=contaminated_paths,
    )

    # Preserve all RNG streams so cached support metadata cannot alter training.
    with _preserve_rng_state():
        prepare_result = prepare_tailguard_metadata(
            model,
            full_train_eval_dataloader,
            device,
            args.tg_prepare_dir,
            args,
            manifest_path=manifest_path,
            manifest_requested_path=args.diag_manifest_path,
        )
    torch.set_rng_state(post_model_rng_state['torch'].detach().cpu().byte())
    if torch.cuda.is_available() and post_model_rng_state['cuda'] is not None:
        torch.cuda.set_rng_state_all([
            value.detach().cpu().byte() for value in post_model_rng_state['cuda']
        ])
    np.random.set_state(post_model_rng_state['numpy'])
    random.setstate(post_model_rng_state['python'])
    print_fn('tailguard prepare: T={}, H={}, head_groups={}'.format(
        prepare_result['summary']['num_tail_candidates'],
        prepare_result['summary']['num_head_candidates'],
        prepare_result['summary']['num_head_groups'],
    ))
    partition_guard_summary = prepare_result['summary'].get('tail_partition_guard', {})
    if bool(partition_guard_summary.get('triggered', False)):
        print_fn(
            'tail partition guard triggered: candidates {} -> {}, cutoff {} -> {}, gap=({}, {})'.format(
                partition_guard_summary.get('num_selected_raw'),
                partition_guard_summary.get('num_selected_final'),
                partition_guard_summary.get('original_cutoff'),
                partition_guard_summary.get('effective_cutoff'),
                partition_guard_summary.get('gap_lower'),
                partition_guard_summary.get('gap_upper'),
            )
        )
    analysis_summary = prepare_result['summary'].get('tail_sampler_analysis_only_summary')
    if analysis_summary is not None and 'selection_precision' in analysis_summary:
        print_fn('tail sampler analysis-only: precision={:.4f}, recall={:.4f}, f1={:.4f}, noisy_proxy_rate={:.4f}'.format(
            float(analysis_summary['selection_precision']),
            float(analysis_summary['selection_recall']),
            float(analysis_summary['selection_f1']),
            float(analysis_summary['selected_noise_proxy_rate']),
        ))
    elif analysis_summary is not None:
        print_fn('tail sampler analysis-only skipped: {}'.format(analysis_summary.get('error', analysis_summary.get('status', 'unknown'))))

    full_metadata_df = prepare_result['full_metadata_df'].copy()
    method_metadata_df = full_metadata_df.drop(columns=['is_contaminated', 'class_name'], errors='ignore')
    stage1_training_scope = args.tg_stage1_training_scope
    if stage1_training_scope == 'h_only':
        h_metadata_df = method_metadata_df.loc[
            method_metadata_df['tail_candidate'].astype(int) == 0
        ].copy().reset_index(drop=True)
        h_index_map = _build_retained_index_map(h_metadata_df, num_classes=len(base_train_data_list))
        stage1_loaders = rebuild_pruned_train_loaders(
            base_train_data_list,
            h_index_map,
            data_root=args.data_path,
            item_list=item_list,
            batch_size=batch_size,
            num_workers=num_workers,
            diag_batch_size=args.diag_batch_size,
            diag_num_workers=args.diag_num_workers,
            contaminated_paths=contaminated_paths,
            sampler=None,
            epoch_num_samples=None,
        )
        train_data_list = stage1_loaders['train_data_list']
        train_data = stage1_loaders['train_data']
        train_dataloader = stage1_loaders['train_dataloader']
        train_eval_dataloader = stage1_loaders['train_eval_dataloader']
    else:
        train_data_list = base_train_data_list
        train_data = full_train_data
        train_dataloader = full_train_dataloader
        train_eval_dataloader = full_train_eval_dataloader
    memory_train_eval_dataloader = full_train_eval_dataloader
    print_fn('stage1 training scope={} train image number:{} / full {}'.format(
        stage1_training_scope,
        len(train_data),
        len(full_train_data),
    ))

    gbps_best_U = None
    gbps_best_SE = None
    gbps_best_iter = None
    gbps_best_noise_evidence = None
    gbps_best_check_count = None
    gbps_check_count = 0
    gbps_trigger_iter = None
    gbps_selected_iter = None
    gbps_has_postprocessed = method_mode == 'core'
    last_eval_iter = None

    it = 0
    while it < total_iters:
        loss_list = []
        reset_loader = False
        for batch in train_dataloader:
            images, _ = batch if len(batch) == 2 else batch[:2]
            loss_value = train_step(
                model,
                trainable,
                optimizer,
                lr_scheduler,
                images,
                current_zero_based_iter=it,
                sample_weight=None,
                grad_clip_norm=float(args.grad_clip_norm),
            )
            loss_list.append(loss_value)
            current_iter = it + 1

            if method_mode != 'core' and gbps_trigger_iter is None and not gbps_has_postprocessed and should_run_check(
                current_iter,
                total_iters,
                float(args.gbps_check_start_ratio),
                float(args.gbps_check_end_ratio),
                int(args.gbps_check_interval),
            ):
                gbps_check_count += 1
                iter_dir = os.path.join(args.diag_save_dir, 'iter_{:05d}'.format(current_iter))
                with _preserve_rng_state():
                    scored_df = score_trainset(
                        model,
                        full_train_eval_dataloader,
                        device,
                        max_ratio=float(args.diag_max_ratio),
                        resize_mask=int(args.diag_resize_mask),
                    )
                    gbps_result = run_tailguard_gbps_iteration(
                        scored_df,
                        prepare_result['head_group_assignments_df'],
                        method_metadata_df,
                        args,
                    )
                t_max = max(1, int(total_iters * float(args.gbps_check_end_ratio)))
                trigger_result = evaluate_gbps_ci_peak_trigger(
                    U_t=gbps_result['gbps_U'],
                    SE_t=gbps_result['gbps_SE'],
                    noise_evidence=gbps_result['global_noise_evidence'],
                    best_U=gbps_best_U,
                    best_SE=gbps_best_SE,
                    best_iter=gbps_best_iter,
                    best_noise_evidence=gbps_best_noise_evidence,
                    best_check_count=gbps_best_check_count,
                    check_count=gbps_check_count,
                    current_iter=current_iter,
                    ci_z=float(args.gbps_ci_z),
                    improve_eps=float(args.gbps_improve_eps),
                    min_checks_before_trigger=int(args.gbps_min_checks_before_trigger),
                    min_checks_after_best=int(args.gbps_min_checks_after_best),
                    min_noise_evidence=float(args.gbps_min_noise_evidence),
                    force_trigger=current_iter >= t_max,
                )
                gbps_best_U = trigger_result['best_U']
                gbps_best_SE = trigger_result['best_SE']
                gbps_best_iter = trigger_result['best_iter']
                gbps_best_noise_evidence = trigger_result['best_noise_evidence']
                gbps_best_check_count = trigger_result['best_check_count']

                summary = dict(gbps_result['summary'])
                summary.update(trigger_result['summary'])
                summary['prepare_summary'] = prepare_result['summary']
                summary['diag_manifest_path'] = manifest_path
                summary['diag_manifest_requested_path'] = args.diag_manifest_path
                summary['diag_has_contamination_labels'] = bool(contaminated_paths is not None)
                save_tailguard_gbps_iteration_artifacts(
                    iter_dir,
                    gbps_result['train_scores_df'],
                    gbps_result['h_group_metrics_df'],
                    gbps_result['h_sample_group_scores_df'],
                    gbps_result['bootstrap_df'],
                    summary,
                )

                if bool(trigger_result['summary']['gbps_improved']):
                    save_tailguard_selected_checkpoint(
                        selected_checkpoint_path,
                        model,
                        optimizer,
                        lr_scheduler,
                        iteration=current_iter,
                        gbps_U=gbps_result['gbps_U'],
                        gbps_SE=gbps_result['gbps_SE'],
                        args=args,
                    )
                    print_fn('tailguard selected checkpoint updated at iter {}: U={:.6f}, SE={:.6f}'.format(
                        current_iter,
                        float(gbps_result['gbps_U']),
                        float(gbps_result['gbps_SE']),
                    ))

                if trigger_result['triggered']:
                    gbps_trigger_iter = current_iter
                    gbps_selected_iter, h_zero_removal_fallback = _resolve_tailguard_trigger_selection(
                        trigger_result
                    )
                    selected_scored_df, selected_score_iter, selected_score_path = resolve_gbps_best_scored_df(
                        gbps_result['train_scores_df'],
                        current_iter,
                        gbps_selected_iter,
                        args.diag_save_dir,
                    )
                    if (
                        (selected_score_iter is None) != (gbps_selected_iter is None)
                        or (
                            selected_score_iter is not None
                            and int(selected_score_iter) != int(gbps_selected_iter)
                        )
                    ):
                        raise RuntimeError(
                            'GBPS selected/score source iteration mismatch: selected={}, score_source={}'.format(
                                gbps_selected_iter,
                                selected_score_iter,
                            )
                        )
                    gbps_trigger_summary = {
                        'stage': 'tailguard_selected',
                        'gbps_trigger_iter': int(gbps_trigger_iter),
                        'gbps_selected_iter': None if gbps_selected_iter is None else int(gbps_selected_iter),
                        'score_source_iter': None if selected_score_iter is None else int(selected_score_iter),
                        'score_source_path': selected_score_path,
                        'selected_checkpoint_path': (
                            None if h_zero_removal_fallback else selected_checkpoint_path
                        ),
                        'checkpoint_rollback_performed': not h_zero_removal_fallback,
                        'gbps_status': trigger_result['status'],
                        'h_zero_removal_fallback': bool(h_zero_removal_fallback),
                        'h_zero_removal_fallback_reason': (
                            'insufficient_head_noise_evidence'
                            if h_zero_removal_fallback else None
                        ),
                        'gbps_prune_mode': args.gbps_prune_mode,
                        'gbps_prune_max_ratio': float(args.gbps_prune_max_ratio),
                        'gbps_prune_ratio': float(args.gbps_prune_ratio),
                    }
                    save_tailguard_trigger_summary(args.diag_save_dir, gbps_trigger_summary)

                    selected_checkpoint_ready = (
                        h_zero_removal_fallback or os.path.isfile(selected_checkpoint_path)
                    )
                    if args.gbps_postprocess_mode == 'remove' and selected_scored_df is not None and selected_checkpoint_ready:
                        analysis_context = _preserve_rng_state() if h_zero_removal_fallback else nullcontext()
                        with analysis_context:
                            head_delete_plan = build_tailguard_head_delete_plan(
                                selected_scored_df,
                                prepare_result['head_group_assignments_df'],
                                method_metadata_df,
                                args,
                                gbps_dir=args.diag_save_dir,
                                selected_iter=selected_score_iter,
                                force_no_delete_reason=(
                                    'insufficient_head_noise_evidence'
                                    if h_zero_removal_fallback else None
                                ),
                            )
                            head_prune_saved = save_tailguard_head_prune_artifacts(
                                args.tg_stage2_dir,
                                head_delete_plan['h_prune_decisions_df'],
                                head_delete_plan['h_prune_group_summary_df'],
                                head_delete_plan['summary'],
                            )
                            selected_scores_df = head_delete_plan['selected_scores_df']
                            tail_df = selected_scores_df.loc[selected_scores_df['tail_candidate'].astype(int) == 1].copy().reset_index(drop=True)
                            if _mode_uses_reconciliation(method_mode):
                                attachment_plan = build_tail_attachment_plan(
                                    tail_df,
                                    head_delete_plan['h_clean_samples'],
                                    prepare_result['grouping_embeddings_payload'],
                                    args,
                                    args.tg_attachment_membership_mode,
                                )
                                geometry = attachment_plan['geometry']
                                conformity_df = attachment_plan['conformity_df']
                                attachment_scores_df = attachment_plan['attachment_scores_df']
                                tail_open_df = attachment_plan['tail_open_df']
                                tail_attached_df = attachment_plan['tail_attached_df']
                                elbow_summary = attachment_plan['attachment_summary']
                                membership_scores_df = attachment_plan['membership_scores_df']
                                membership_calibration_df = attachment_plan['membership_calibration_df']
                                membership_summary = attachment_plan['membership_summary']
                                rgd_distances_df = attachment_plan['rgd_distances_df']
                                rgd_scores_df = attachment_plan['rgd_scores_df']
                                rgd_split_summary = attachment_plan['rgd_split_summary']
                                rgd_bic_candidates_df = attachment_plan['rgd_bic_candidates_df']
                                tail_head_normal_df = tail_attached_df.copy().sort_values(
                                    'sample_idx', kind='mergesort'
                                ).reset_index(drop=True)
                                tail_head_noise_df = tail_attached_df.iloc[0:0].copy()
                                attachment_saved = save_tailguard_attachment_artifacts(
                                    args.tg_attachment_dir,
                                    geometry,
                                    conformity_df,
                                    attachment_scores_df,
                                    elbow_summary,
                                    tail_open_df,
                                    tail_attached_df,
                                    tail_head_normal_df,
                                    tail_head_noise_df,
                                    membership_scores_df=membership_scores_df,
                                    membership_calibration_df=membership_calibration_df,
                                    membership_summary=membership_summary,
                                    rgd_distances_df=rgd_distances_df,
                                    rgd_scores_df=rgd_scores_df,
                                    rgd_split_summary=rgd_split_summary,
                                    rgd_bic_candidates_df=rgd_bic_candidates_df,
                                )
                                stage2_plan = build_tailguard_stage2_plan(
                                    head_delete_plan['h_clean_samples'],
                                    head_delete_plan['h_removed_samples'],
                                    tail_open_df,
                                    tail_head_normal_df,
                                    tail_head_noise_df,
                                    all_samples_df=method_metadata_df,
                                )
                            elif method_mode == 'h_raw_e':
                                tail_open_df, tail_head_normal_df, tail_head_noise_df = _build_raw_tail_roles(tail_df)
                                stage2_plan = build_tailguard_stage2_plan(
                                    head_delete_plan['h_clean_samples'],
                                    head_delete_plan['h_removed_samples'],
                                    tail_open_df,
                                    tail_head_normal_df,
                                    tail_head_noise_df,
                                    all_samples_df=method_metadata_df,
                                )
                            elif method_mode == 'h_only':
                                tail_open_df = tail_df.copy()
                                tail_head_normal_df = _empty_tail_affiliated_frame(tail_df)
                                tail_head_noise_df = tail_head_normal_df.copy()
                                stage2_plan = _build_h_only_stage2_plan(
                                    head_delete_plan,
                                    tail_df,
                                    method_metadata_df,
                                    num_classes=len(base_train_data_list),
                                )
                            else:
                                raise ValueError('unexpected cleanup-capable method mode: {}'.format(method_mode))
                            _validate_stage2_partition(stage2_plan, method_metadata_df, tail_df, method_mode)
                            stage2_plan['summary'].update({
                                'method_mode': method_mode,
                                'reconciliation_performed': _mode_uses_reconciliation(method_mode),
                                'enhancement_performed': _mode_uses_enhancement(method_mode),
                                'num_tail_candidates_removed': int(len(tail_head_noise_df)),
                                'tail_candidate_protection_enforced': True,
                                'h_zero_removal_fallback': bool(h_zero_removal_fallback),
                            })
                            stage2_saved = save_tailguard_stage2_artifacts(
                                args.tg_stage2_dir,
                                stage2_plan['retained_samples'],
                                stage2_plan['removed_samples'],
                                stage2_plan['summary'],
                            )
                            denoising_diagnostics_summary, denoising_diagnostics_by_class = build_tailguard_denoising_diagnostics(
                                prepare_result['analysis_metadata_df'],
                                stage2_plan['retained_samples'],
                                stage2_plan['removed_samples'],
                                h_removed_samples_df=head_delete_plan['h_removed_samples'],
                                tail_head_noise_samples_df=tail_head_noise_df,
                                manifest_path=manifest_path,
                                manifest_requested_path=args.diag_manifest_path,
                                label_source=prepare_result['saved'].get('tailguard_train_analysis_metadata_csv') if prepare_result['saved'] else None,
                            )
                            denoising_diagnostics_artifacts = save_tailguard_denoising_diagnostics(
                                args.tg_root_dir,
                                denoising_diagnostics_summary,
                                denoising_diagnostics_by_class,
                            )
                            if _mode_uses_enhancement(method_mode):
                                cls_embeddings_payload = prepare_result['cls_embeddings_payload']
                                if cls_embeddings_payload is None:
                                    raise ValueError(
                                        'TailGuard enhancement requires --tg_save_cls_embeddings'
                                    )
                                pseudoclass_registry = build_tailguard_pseudoclass_registry(
                                    head_delete_plan['h_clean_samples'],
                                    tail_head_normal_df,
                                    tail_open_df,
                                    stage2_plan['retained_samples'],
                                    stage2_plan['removed_samples'],
                                    cls_embeddings_payload,
                                )
                                pseudoclass_artifacts = save_tailguard_pseudoclass_artifacts(
                                    os.path.join(args.tg_root_dir, 'pseudoclasses'),
                                    pseudoclass_registry['members_df'],
                                    pseudoclass_registry['classes_df'],
                                    pseudoclass_registry['tail_edges_df'],
                                    pseudoclass_registry['summary'],
                                    metadata={'dataset_provenance': args.dataset_provenance},
                                )
                                pseudoclass_report, pseudoclass_predictions, pseudoclass_summary_df, pseudoclass_contingency_df = (
                                    build_tailguard_pseudoclass_report(
                                        pseudoclass_registry['members_df'],
                                        pseudoclass_registry['classes_df'],
                                        cls_embeddings_payload,
                                        prepare_result['analysis_metadata_df'],
                                    )
                                )
                                pseudoclass_report_artifacts = save_tailguard_pseudoclass_report_artifacts(
                                    args.tg_root_dir,
                                    pseudoclass_report,
                                    pseudoclass_predictions,
                                    pseudoclass_summary_df,
                                    pseudoclass_contingency_df,
                                )
                                pseudoclass_status = 'completed'
                                print_fn('tailguard pseudo-classes: total={}, head={}, tail={}'.format(
                                    pseudoclass_registry['summary']['num_pseudo_classes'],
                                    pseudoclass_registry['summary']['num_head_pseudo_classes'],
                                    pseudoclass_registry['summary']['num_tail_pseudo_classes'],
                                ))

                        if not h_zero_removal_fallback:
                            checkpoint = load_tailguard_selected_checkpoint(
                                selected_checkpoint_path,
                                model,
                                optimizer=optimizer,
                                scheduler=lr_scheduler,
                                map_location=device,
                                restore_rng=True,
                            )
                            it = int(checkpoint['iteration'])
                            rebuild_result = rebuild_pruned_train_loaders(
                                base_train_data_list,
                                stage2_plan['retained_index_map'],
                                data_root=args.data_path,
                                item_list=item_list,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                diag_batch_size=args.diag_batch_size,
                                diag_num_workers=args.diag_num_workers,
                                contaminated_paths=contaminated_paths,
                                sampler=None,
                                epoch_num_samples=None,
                            )
                            train_data_list = rebuild_result['train_data_list']
                            train_data = rebuild_result['train_data']
                            train_dataloader = rebuild_result['train_dataloader']
                            train_eval_dataloader = rebuild_result['train_eval_dataloader']
                            memory_train_eval_dataloader = train_eval_dataloader
                            reset_loader = True
                        gbps_has_postprocessed = True
                        print_fn('tailguard trigger iter {} selected iter {}: H prune mode {}, stage2 retained {}, removed {}'.format(
                            gbps_trigger_iter,
                            gbps_selected_iter,
                            head_delete_plan['prune_mode'],
                            stage2_plan['summary']['num_stage2_retained'],
                            stage2_plan['summary']['num_stage2_removed'],
                        ))
                        if reset_loader:
                            break
                    else:
                        gbps_has_postprocessed = True
                        print_fn('tailguard trigger iter {} did not run removal; continuing without Stage 2 rebuild'.format(gbps_trigger_iter))

            if current_iter % int(args.eval_interval) == 0:
                final_eval_summary = _evaluate_current_model(model, test_data_list, item_list, device, args)
                last_eval_iter = int(current_iter)
                model.train()

            it += 1
            if it >= total_iters:
                break

        if reset_loader:
            continue
        if len(loss_list) > 0:
            print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
        else:
            break

    if denoising_diagnostics_artifacts is None:
        if method_mode == 'core':
            cleanup_reason = 'method_mode_core'
        elif gbps_trigger_iter is None:
            cleanup_reason = 'gbps_not_triggered'
        elif args.gbps_postprocess_mode != 'remove':
            cleanup_reason = 'gbps_postprocess_mode_{}'.format(args.gbps_postprocess_mode)
        elif not os.path.isfile(selected_checkpoint_path):
            cleanup_reason = 'selected_checkpoint_unavailable'
        else:
            cleanup_reason = 'stage2_plan_unavailable'
        denoising_diagnostics_summary, denoising_diagnostics_by_class = build_tailguard_denoising_diagnostics(
            prepare_result['analysis_metadata_df'],
            method_metadata_df,
            method_metadata_df.iloc[0:0].copy(),
            manifest_path=manifest_path,
            manifest_requested_path=args.diag_manifest_path,
            label_source=prepare_result['saved'].get('tailguard_train_analysis_metadata_csv') if prepare_result['saved'] else None,
            cleanup_status='not_performed',
            cleanup_reason=cleanup_reason,
        )
        denoising_diagnostics_artifacts = save_tailguard_denoising_diagnostics(
            args.tg_root_dir,
            denoising_diagnostics_summary,
            denoising_diagnostics_by_class,
        )

    if pseudoclass_status != 'completed':
        if not _mode_uses_enhancement(method_mode):
            pseudoclass_reason = 'method_mode_{}_disables_enhancement'.format(method_mode)
        else:
            pseudoclass_reason = 'cleanup_status_{}'.format(
                denoising_diagnostics_summary['cleanup_status']
            )

    if last_eval_iter != int(it):
        final_eval_summary = _evaluate_current_model(model, test_data_list, item_list, device, args)
        last_eval_iter = int(it)
        model.train()

    checkpoint_path = save_train_checkpoint(
        args.save_dir,
        args.save_name,
        model,
        iteration=it,
        args=args,
        final_eval_summary=final_eval_summary,
    )
    print_fn('saved final model checkpoint to {} before memory evaluation'.format(checkpoint_path))

    memory_system = None
    memory_enabled = bool(args.tg_memory_enable) and _mode_uses_enhancement(method_mode)
    memory_status = 'disabled' if not memory_enabled else 'skipped_pseudoclasses_unavailable'
    if memory_enabled and pseudoclass_registry is not None:
        memory_system = build_tailguard_pseudoclass_memory_system(
            model,
            memory_train_eval_dataloader,
            device,
            pseudoclass_registry['members_df'],
            pseudoclass_registry['classes_df'],
            args,
        )
        print_fn('tailguard memory: built {} class-wise tail banks across {} pseudo-classes'.format(
            memory_system['num_tail_memory_banks'],
            memory_system['num_pseudo_classes'],
        ))
        score_df, per_class_metrics, memory_eval_summary, metrics_by_mode = run_pseudoclass_memory_evaluation(
            model,
            test_data_list,
            item_list,
            memory_system,
            args,
            device,
        )
        memory_metadata = {
            'dataset_provenance': args.dataset_provenance,
            'checkpoint_path': args.checkpoint_path,
            'encoder_name': checkpoint_metadata['encoder_name'] if checkpoint_metadata is not None else args.encoder_name,
            'image_size': image_size,
            'crop_size': crop_size,
            'num_pseudo_classes': int(memory_system['num_pseudo_classes']),
            'num_head_pseudo_classes': int(memory_system['num_head_pseudo_classes']),
            'num_tail_pseudo_classes': int(memory_system['num_tail_pseudo_classes']),
            'num_tail_memory_banks': int(memory_system['num_tail_memory_banks']),
            'tg_memory_fusion_lambda': float(args.tg_memory_fusion_lambda),
            'tg_memory_topk_ratio': float(args.tg_memory_topk_ratio),
            'tg_memory_route_margin_threshold': float(args.tg_memory_route_margin_threshold),
            'tg_memory_min_class_members': int(args.tg_memory_min_class_members),
            'tg_mem_max_patches_per_class': int(args.tg_mem_max_patches_per_class),
        }
        memory_saved = save_tailguard_memory_artifacts(
            args.tg_memory_dir,
            memory_system,
            score_df,
            per_class_metrics,
            memory_eval_summary,
            memory_metadata,
            metrics_by_mode=metrics_by_mode,
        )
        memory_status = 'completed'
        print_fn('saved tailguard memory artifacts to {}'.format(args.tg_memory_dir))

    total_time = time.time() - train_start_time
    time_per_iter = total_time / max(1, total_iters)
    iters_per_sec = total_iters / max(total_time, 1e-12)
    samples_per_sec = batch_size / max(time_per_iter, 1e-12)

    print_fn('Training finished. Total time: {:.2f}s ({:.2f} min), {:.4f} s/iter'.format(total_time, total_time / 60.0, time_per_iter))
    if memory_eval_summary is not None:
        print_fn('Final TailGuard fused evaluation:')
        _print_eval_summary(memory_eval_summary, print_fn)
    print_fn('Training efficiency:')
    print_fn('  total_time_s      {:.2f}'.format(total_time))
    print_fn('  total_iters       {}'.format(total_iters))
    print_fn('  batch_size        {}'.format(batch_size))
    print_fn('  time_per_iter_s   {:.4f}'.format(time_per_iter))
    print_fn('  iters_per_sec     {:.4f}'.format(iters_per_sec))
    print_fn('  samples_per_sec   {:.2f}'.format(samples_per_sec))

    tailguard_summary = {
        'schema_version': TAILGUARD_CONFIG_SCHEMA_VERSION,
        'method_name': TAILGUARD_METHOD_NAME,
        'config_profile': TAILGUARD_CONFIG_PROFILE,
        'method_mode': method_mode,
        'resolved_method_config': args.tailguard_resolved_method_config,
        'dataset_provenance': args.dataset_provenance,
        'checkpoint_path': checkpoint_path,
        'input_checkpoint_path': args.checkpoint_path,
        'selected_checkpoint_path': (
            None
            if gbps_trigger_summary is not None
            and bool(gbps_trigger_summary.get('h_zero_removal_fallback'))
            else selected_checkpoint_path if os.path.isfile(selected_checkpoint_path) else None
        ),
        'prepare_summary': prepare_result['summary'],
        'gbps_trigger_summary': gbps_trigger_summary,
        'head_prune_summary': None if stage2_plan is None else head_delete_plan['summary'],
        'head_prune_artifacts': head_prune_saved,
        'stage2_summary': None if stage2_plan is None else stage2_plan['summary'],
        'attachment_artifacts': attachment_saved,
        'attachment_replay_config_path': getattr(args, 'tg_attachment_replay_config_path', None),
        'stage2_artifacts': stage2_saved,
        'denoising_diagnostics_artifacts': denoising_diagnostics_artifacts,
        'denoising_diagnostics_summary': {
            'cleanup_status': denoising_diagnostics_summary['cleanup_status'],
            'cleanup_reason': denoising_diagnostics_summary['cleanup_reason'],
            'decision_counts': denoising_diagnostics_summary['decision_counts'],
            'contamination_counts': denoising_diagnostics_summary['contamination_counts'],
            'contamination_metrics': denoising_diagnostics_summary['contamination_metrics'],
        },
        'pseudo_class_status': pseudoclass_status,
        'pseudo_class_reason': pseudoclass_reason,
        'pseudo_class_artifacts': pseudoclass_artifacts,
        'pseudo_class_report_artifacts': pseudoclass_report_artifacts,
        'pseudo_class_summary': None if pseudoclass_registry is None else {
            'num_pseudo_classes': pseudoclass_registry['summary']['num_pseudo_classes'],
            'num_head_pseudo_classes': pseudoclass_registry['summary']['num_head_pseudo_classes'],
            'num_tail_pseudo_classes': pseudoclass_registry['summary']['num_tail_pseudo_classes'],
            'num_tail_singleton_classes': pseudoclass_registry['summary']['num_tail_singleton_classes'],
        },
        'final_eval_summary': final_eval_summary,
        'memory_status': memory_status,
        'memory_config': {
            'tg_memory_fusion_lambda': float(args.tg_memory_fusion_lambda),
            'tg_memory_topk_ratio': float(args.tg_memory_topk_ratio),
            'tg_memory_route_margin_threshold': float(args.tg_memory_route_margin_threshold),
            'tg_memory_min_class_members': int(args.tg_memory_min_class_members),
            'tg_mem_max_patches_per_class': int(args.tg_mem_max_patches_per_class),
        },
        'memory_eval_summary': memory_eval_summary,
        'memory_artifacts': memory_saved,
        'num_pseudo_classes': 0 if memory_system is None else int(memory_system['num_pseudo_classes']),
        'num_head_pseudo_classes': 0 if memory_system is None else int(memory_system['num_head_pseudo_classes']),
        'num_tail_pseudo_classes': 0 if memory_system is None else int(memory_system['num_tail_pseudo_classes']),
        'num_tail_memory_banks': 0 if memory_system is None else int(memory_system['num_tail_memory_banks']),
        'total_time_s': float(total_time),
        'time_per_iter_s': float(time_per_iter),
        'iters_per_sec': float(iters_per_sec),
        'samples_per_sec': float(samples_per_sec),
    }
    summary_path = save_tailguard_summary(args.tg_root_dir, tailguard_summary)
    print_fn('saved tailguard summary to {}'.format(summary_path))


def build_parser(default_dataset_profile='mvtec'):
    default_profile = get_dataset_profile(default_dataset_profile)
    parser = argparse.ArgumentParser(description='Standalone TailGuard runner for MVTec-compatible datasets')
    parser.add_argument('--dataset_profile', type=str, choices=dataset_profile_names(), default=default_profile.name)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--diag_save_dir', type=str, default='gbps')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--total_iters', type=int, default=DEFAULT_TOTAL_ITERS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--image_size', type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument('--crop_size', type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument('--encoder_name', type=str, default=DEFAULT_ENCODER_NAME)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--final_lr', type=float, default=2e-4)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--max_ratio', type=float, default=0.01)
    parser.add_argument('--resize_mask', type=int, default=256)
    parser.add_argument('--diag_batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--diag_num_workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--diag_max_ratio', type=float, default=0.01)
    parser.add_argument('--diag_resize_mask', type=int, default=256)
    parser.add_argument('--diag_manifest_path', type=str, default=None)

    parser.add_argument(
        '--tg_method_mode',
        type=str,
        default=TAILGUARD_FINAL_DEFAULTS['tg_method_mode'],
        choices=TAILGUARD_METHOD_MODES,
        help='Dependency-aware paper variant; full is the canonical TailGuard method.',
    )
    parser.add_argument('--tailsampler_type', type=str, default=TAILGUARD_FINAL_DEFAULTS['tailsampler_type'], choices=['tail', 'adaptive', 'adaptive_trim_mode'])
    parser.add_argument('--tailsampler_th_type', type=str, default=TAILGUARD_FINAL_DEFAULTS['tailsampler_th_type'])
    parser.add_argument('--tailsampler_vote_type', type=str, default=TAILGUARD_FINAL_DEFAULTS['tailsampler_vote_type'])
    parser.add_argument('--tailsampler_percentile', type=float, default=TAILGUARD_FINAL_DEFAULTS['tailsampler_percentile'])
    parser.add_argument('--tailsampler_plateau_gap_guard', dest='tailsampler_plateau_gap_guard', action='store_true', default=TAILGUARD_FINAL_DEFAULTS['tailsampler_plateau_gap_guard'])
    parser.add_argument('--no-tailsampler_plateau_gap_guard', dest='tailsampler_plateau_gap_guard', action='store_false')
    parser.add_argument('--tailsampler_embedding_source', type=str, default=TAILGUARD_FINAL_DEFAULTS['tailsampler_embedding_source'], choices=['encoder', 'encoder_cls'])
    parser.add_argument('--tailsampler_gt_mode', type=str, default=TAILGUARD_FINAL_DEFAULTS['tailsampler_gt_mode'], choices=['true_class', 'dataset_rule'])
    parser.add_argument('--tailsampler_tail_count_thr', type=int, default=TAILGUARD_FINAL_DEFAULTS['tailsampler_tail_count_thr'])
    parser.add_argument('--tailsampler_dataset_name', type=str, default=TAILGUARD_FINAL_DEFAULTS['tailsampler_dataset_name'])

    parser.add_argument('--gbps_check_start_ratio', type=float, default=TAILGUARD_FINAL_DEFAULTS['gbps_check_start_ratio'])
    parser.add_argument('--gbps_check_end_ratio', type=float, default=TAILGUARD_FINAL_DEFAULTS['gbps_check_end_ratio'])
    parser.add_argument('--gbps_check_interval', type=int, default=TAILGUARD_FINAL_DEFAULTS['gbps_check_interval'])
    parser.add_argument('--gbps_grouping_method', type=str, default=TAILGUARD_FINAL_DEFAULTS['gbps_grouping_method'])
    parser.add_argument('--gbps_num_groups', type=int, default=TAILGUARD_FINAL_DEFAULTS['gbps_num_groups'])
    parser.add_argument('--gbps_auto_k', dest='gbps_auto_k', action='store_true', default=TAILGUARD_FINAL_DEFAULTS['gbps_auto_k'])
    parser.add_argument('--no-gbps_auto_k', dest='gbps_auto_k', action='store_false')
    parser.add_argument('--gbps_k_candidates', type=str, default=TAILGUARD_FINAL_DEFAULTS['gbps_k_candidates'])
    parser.add_argument('--gbps_pca_dim', type=int, default=TAILGUARD_FINAL_DEFAULTS['gbps_pca_dim'])
    parser.add_argument('--gbps_min_group_size', type=int, default=TAILGUARD_FINAL_DEFAULTS['gbps_min_group_size'])
    parser.add_argument('--gbps_embedding_source', type=str, default=TAILGUARD_FINAL_DEFAULTS['gbps_embedding_source'])
    parser.add_argument('--gbps_tau_bic', type=float, default=TAILGUARD_FINAL_DEFAULTS['gbps_tau_bic'])
    parser.add_argument('--gbps_tau_sep', type=float, default=TAILGUARD_FINAL_DEFAULTS['gbps_tau_sep'])
    parser.add_argument('--gbps_tau_conf', type=float, default=TAILGUARD_FINAL_DEFAULTS['gbps_tau_conf'])
    parser.add_argument('--gbps_pi_min', type=float, default=TAILGUARD_FINAL_DEFAULTS['gbps_pi_min'])
    parser.add_argument('--gbps_pi_max', type=float, default=TAILGUARD_FINAL_DEFAULTS['gbps_pi_max'])
    parser.add_argument('--gbps_gate_mode', type=str, default=TAILGUARD_FINAL_DEFAULTS['gbps_gate_mode'])
    parser.add_argument('--gbps_min_noise_evidence', type=float, default=TAILGUARD_FINAL_DEFAULTS['gbps_min_noise_evidence'])
    parser.add_argument('--gbps_postprocess_mode', type=str, default=TAILGUARD_FINAL_DEFAULTS['gbps_postprocess_mode'], choices=['none', 'remove'])
    parser.add_argument('--gbps_prune_mode', type=str, default=TAILGUARD_FINAL_DEFAULTS['gbps_prune_mode'], choices=TAILGUARD_PRUNE_MODES)
    parser.add_argument('--gbps_prune_max_ratio', type=float, default=TAILGUARD_FINAL_DEFAULTS['gbps_prune_max_ratio'])
    parser.add_argument('--gbps_prune_stable_window', type=int, default=TAILGUARD_FINAL_DEFAULTS['gbps_prune_stable_window'])
    parser.add_argument('--gbps_prune_stable_min_observations', type=int, default=TAILGUARD_FINAL_DEFAULTS['gbps_prune_stable_min_observations'])
    parser.add_argument('--gbps_prune_min_active_ratio', type=float, default=TAILGUARD_FINAL_DEFAULTS['gbps_prune_min_active_ratio'])
    parser.add_argument('--gbps_prune_ratio', type=float, default=TAILGUARD_FINAL_DEFAULTS['gbps_prune_ratio'])
    parser.add_argument('--gbps_min_keep_per_group', type=int, default=TAILGUARD_FINAL_DEFAULTS['gbps_min_keep_per_group'])
    parser.add_argument('--gbps_bootstrap_B', type=int, default=TAILGUARD_FINAL_DEFAULTS['gbps_bootstrap_B'])
    parser.add_argument('--gbps_ci_z', type=float, default=TAILGUARD_FINAL_DEFAULTS['gbps_ci_z'])
    parser.add_argument('--gbps_improve_eps', type=float, default=TAILGUARD_FINAL_DEFAULTS['gbps_improve_eps'])
    parser.add_argument('--gbps_min_checks_before_trigger', type=int, default=TAILGUARD_FINAL_DEFAULTS['gbps_min_checks_before_trigger'])
    parser.add_argument('--gbps_min_checks_after_best', type=int, default=TAILGUARD_FINAL_DEFAULTS['gbps_min_checks_after_best'])

    parser.add_argument('--tg_tail_embedding_source', type=str, default=TAILGUARD_FINAL_DEFAULTS['tg_tail_embedding_source'], choices=['encoder', 'encoder_cls'])
    parser.add_argument('--tg_group_embedding_source', type=str, default=TAILGUARD_FINAL_DEFAULTS['tg_group_embedding_source'], choices=['encoder', 'encoder_cls'])
    parser.add_argument('--tg_stage1_training_scope', type=str, default=TAILGUARD_FINAL_DEFAULTS['tg_stage1_training_scope'], choices=['all_samples'])
    parser.add_argument('--tg_save_cls_embeddings', dest='tg_save_cls_embeddings', action='store_true', default=TAILGUARD_FINAL_DEFAULTS['tg_save_cls_embeddings'])
    parser.add_argument('--no-tg_save_cls_embeddings', dest='tg_save_cls_embeddings', action='store_false')
    parser.add_argument('--tg_save_grouping_embeddings', dest='tg_save_grouping_embeddings', action='store_true', default=TAILGUARD_FINAL_DEFAULTS['tg_save_grouping_embeddings'])
    parser.add_argument('--no-tg_save_grouping_embeddings', dest='tg_save_grouping_embeddings', action='store_false')
    parser.add_argument('--tg_save_tailsampler_analysis', dest='tg_save_tailsampler_analysis', action='store_true', default=TAILGUARD_FINAL_DEFAULTS['tg_save_tailsampler_analysis'])
    parser.add_argument('--no-tg_save_tailsampler_analysis', dest='tg_save_tailsampler_analysis', action='store_false')
    parser.add_argument('--tg_save_tailsampler_analysis_details', dest='tg_save_tailsampler_analysis_details', action='store_true', default=TAILGUARD_FINAL_DEFAULTS['tg_save_tailsampler_analysis_details'])
    parser.add_argument('--no-tg_save_tailsampler_analysis_details', dest='tg_save_tailsampler_analysis_details', action='store_false')
    parser.add_argument('--tg_default_adaptive_angle', type=float, default=TAILGUARD_FINAL_DEFAULTS['tg_default_adaptive_angle'])
    parser.add_argument('--tg_min_clean_group_size', type=int, default=TAILGUARD_FINAL_DEFAULTS['tg_min_clean_group_size'])
    parser.add_argument('--tg_attachment_membership_mode', type=str, default=TAILGUARD_FINAL_DEFAULTS['tg_attachment_membership_mode'], choices=['rgd'])
    parser.add_argument('--tg_elbow_min_segment', type=int, default=TAILGUARD_FINAL_DEFAULTS['tg_elbow_min_segment'])
    parser.add_argument('--tg_rgd_split_mode', type=str, default=TAILGUARD_FINAL_DEFAULTS['tg_rgd_split_mode'], choices=['segmented_bic', 'largest_positive_gap'])
    parser.add_argument('--tg_stable_window', type=int, default=TAILGUARD_FINAL_DEFAULTS['tg_stable_window'])
    parser.add_argument('--tg_stable_min_observations', type=int, default=TAILGUARD_FINAL_DEFAULTS['tg_stable_min_observations'])
    parser.add_argument('--tg_t_attached_noise_p_high_thr', type=float, default=TAILGUARD_FINAL_DEFAULTS['tg_t_attached_noise_p_high_thr'])
    parser.add_argument('--tg_memory_enable', dest='tg_memory_enable', action='store_true', default=TAILGUARD_FINAL_DEFAULTS['tg_memory_enable'])
    parser.add_argument('--no-tg_memory_enable', dest='tg_memory_enable', action='store_false')
    parser.add_argument('--tg_memory_fusion_lambda', type=float, default=TAILGUARD_FINAL_DEFAULTS['tg_memory_fusion_lambda'])
    parser.add_argument('--tg_memory_topk_ratio', type=float, default=TAILGUARD_FINAL_DEFAULTS['tg_memory_topk_ratio'])
    parser.add_argument('--tg_memory_route_margin_threshold', type=float, default=TAILGUARD_FINAL_DEFAULTS['tg_memory_route_margin_threshold'])
    parser.add_argument('--tg_memory_min_class_members', type=int, default=TAILGUARD_FINAL_DEFAULTS['tg_memory_min_class_members'])
    parser.add_argument('--tg_mem_chunk_size', type=int, default=TAILGUARD_FINAL_DEFAULTS['tg_mem_chunk_size'])
    parser.add_argument('--tg_mem_max_patches_per_class', type=int, default=TAILGUARD_FINAL_DEFAULTS['tg_mem_max_patches_per_class'])
    return parser


def resolve_dataset_profile_args(parsed_args):
    profile = get_dataset_profile(parsed_args.dataset_profile)
    if parsed_args.data_path is None:
        parsed_args.data_path = profile.default_data_path
    if parsed_args.save_name is None:
        parsed_args.save_name = 'vitill_{}_tailguard'.format(profile.name)
    parsed_args.dataset_profile = profile.name
    parsed_args.dataset_item_list = list(profile.item_list)
    parsed_args.dataset_layout = profile.layout
    parsed_args.dataset_provenance = profile_provenance(profile, parsed_args.data_path)
    return profile


def main(default_dataset_profile='mvtec', argv=None, required_dataset_profile=None):
    global args, device, print_fn

    parser = build_parser(default_dataset_profile)
    args = parser.parse_args(argv)
    profile = resolve_dataset_profile_args(args)
    if required_dataset_profile is not None and profile.name != required_dataset_profile:
        parser.error('this launcher requires --dataset_profile {}'.format(required_dataset_profile))
    if not os.path.isdir(args.data_path):
        parser.error('dataset directory does not exist: {}'.format(args.data_path))
    expected_memory = _mode_uses_enhancement(args.tg_method_mode)
    if expected_memory and not args.tg_save_cls_embeddings:
        parser.error('--tg_save_cls_embeddings is required when enhancement is enabled')
    if args.tg_method_mode in {'core', 'h_only', 'full_no_e'} and bool(args.tg_save_cls_embeddings):
        parser.error('--tg_method_mode {} requires --no-tg_save_cls_embeddings'.format(args.tg_method_mode))
    if bool(args.tg_memory_enable) != expected_memory:
        parser.error(
            '--tg_method_mode {} requires {}'.format(
                args.tg_method_mode,
                '--tg_memory_enable' if expected_memory else '--no-tg_memory_enable',
            )
        )
    if args.tg_method_mode != 'core' and args.gbps_postprocess_mode != 'remove':
        parser.error('--tg_method_mode {} requires --gbps_postprocess_mode remove'.format(args.tg_method_mode))
    if args.tg_method_mode == 'core' and args.gbps_postprocess_mode != 'none':
        parser.error('--tg_method_mode core requires --gbps_postprocess_mode none')
    if args.tg_method_mode == 'core' and float(args.gbps_prune_max_ratio) != 0.0:
        parser.error('--tg_method_mode core requires --gbps_prune_max_ratio 0.0')
    if args.tg_method_mode in {'full_no_e', 'full'} and float(args.tg_t_attached_noise_p_high_thr) != 1.0:
        parser.error('the non-destructive P module requires --tg_t_attached_noise_p_high_thr 1.0')
    if args.tg_stage1_training_scope != 'all_samples':
        parser.error('dependency-aware TailGuard modes require --tg_stage1_training_scope all_samples')
    if not 0.0 <= float(args.gbps_prune_max_ratio) <= 1.0:
        parser.error('--gbps_prune_max_ratio must be in [0, 1]')
    if not 0.0 <= float(args.gbps_prune_ratio) <= 1.0:
        parser.error('--gbps_prune_ratio must be in [0, 1]')
    if int(args.gbps_prune_stable_window) < 1:
        parser.error('--gbps_prune_stable_window must be at least 1')
    if int(args.gbps_prune_stable_min_observations) < 1:
        parser.error('--gbps_prune_stable_min_observations must be at least 1')
    if not 0.0 <= float(args.gbps_prune_min_active_ratio) < 1.0:
        parser.error('--gbps_prune_min_active_ratio must be in [0, 1)')

    args.tailguard_config_schema_version = TAILGUARD_CONFIG_SCHEMA_VERSION
    args.tailguard_method_name = TAILGUARD_METHOD_NAME
    args.tailguard_config_profile = TAILGUARD_CONFIG_PROFILE
    args.tailguard_resolved_method_config = resolved_tailguard_method_config(args)

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    args.tg_root_dir = os.path.join(args.save_dir, args.save_name, 'tailguard')
    args.tg_prepare_dir = os.path.join(args.tg_root_dir, 'prepare')
    args.tg_attachment_dir = os.path.join(args.tg_root_dir, 'attachment')
    args.tg_stage2_dir = os.path.join(args.tg_root_dir, 'stage2')
    args.tg_memory_dir = os.path.join(args.tg_root_dir, 'memory')
    args.tg_checkpoint_dir = os.path.join(args.tg_root_dir, 'checkpoints')
    if not os.path.isabs(args.diag_save_dir):
        args.diag_save_dir = os.path.join(args.tg_root_dir, args.diag_save_dir)
    os.makedirs(args.diag_save_dir, exist_ok=True)
    os.makedirs(args.tg_checkpoint_dir, exist_ok=True)
    args.tg_attachment_replay_config_path = save_tailguard_attachment_replay_config(
        args.tg_root_dir,
        {
            'schema_version': TAILGUARD_CONFIG_SCHEMA_VERSION,
            'method_name': TAILGUARD_METHOD_NAME,
            'config_profile': TAILGUARD_CONFIG_PROFILE,
            'resolved_method_config': args.tailguard_resolved_method_config,
            'dataset_provenance': args.dataset_provenance,
            'tg_attachment_membership_mode': args.tg_attachment_membership_mode,
            'tg_min_clean_group_size': args.tg_min_clean_group_size,
            'tg_elbow_min_segment': args.tg_elbow_min_segment,
            'tg_rgd_split_mode': args.tg_rgd_split_mode,
            'tg_stable_window': args.tg_stable_window,
            'tg_stable_min_observations': args.tg_stable_min_observations,
            'tg_t_attached_noise_p_high_thr': args.tg_t_attached_noise_p_high_thr,
            'gbps_prune_mode': args.gbps_prune_mode,
            'gbps_prune_max_ratio': args.gbps_prune_max_ratio,
            'gbps_prune_stable_window': args.gbps_prune_stable_window,
            'gbps_prune_stable_min_observations': args.gbps_prune_stable_min_observations,
            'gbps_prune_min_active_ratio': args.gbps_prune_min_active_ratio,
            'gbps_prune_ratio': args.gbps_prune_ratio,
            'gbps_min_keep_per_group': args.gbps_min_keep_per_group,
            'tg_gbps_dir_relpath': os.path.relpath(args.diag_save_dir, args.tg_root_dir),
            'tg_gbps_dir': args.diag_save_dir,
        },
    )
    device = 'cuda:{}'.format(args.gpus) if torch.cuda.is_available() else 'cpu'
    print_fn(device)
    train(profile.item_list)
    return args


if __name__ == '__main__':
    main()
