"""End-to-end training pipeline for TailGuard."""

import os
import random
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tailguard.engine.trainer import (
    build_datasets,
    build_model,
    build_optimizer_and_scheduler,
    build_train_dataloader,
    build_train_eval_dataloader,
    evaluate_model,
    rebuild_pruned_train_loaders,
    save_train_checkpoint,
    score_trainset,
    setup_seed,
    should_run_check,
    train_step,
)
from tailguard.method.dhp_utils import evaluate_gbps_ci_peak_trigger, resolve_gbps_best_scored_df
from tailguard.reporting.artifacts import (
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
from tailguard.method.preparation import prepare_tailguard_metadata
from tailguard.method.registry import build_tailguard_pseudoclass_registry
from tailguard.config import (
    TAILGUARD_CONFIG_PROFILE,
    TAILGUARD_CONFIG_SCHEMA_VERSION,
    TAILGUARD_METHOD_NAME,
    variant_uses_enhancement,
    variant_uses_reconciliation,
)
from tailguard.data.manifests import load_injected_manifest

warnings = __import__('warnings')
warnings.filterwarnings('ignore')


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


def _empty_tail_affiliated_frame(tail_df):
    empty = tail_df.iloc[0:0].copy()
    if 'best_group_id' not in empty.columns:
        empty['best_group_id'] = pd.Series(dtype=int)
    return empty


def _build_dhp_stage2_plan(head_delete_plan, tail_df, all_samples_df, num_classes):
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
        raise ValueError('DHP Stage 2 partitions must be disjoint and cover all samples')
    if not tail_ids.issubset(retained_ids):
        raise ValueError('DHP must retain every TailSampler candidate')
    summary = {
        'variant': 'dhp',
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


def _validate_stage2_partition(stage2_plan, all_samples_df, tail_df, variant):
    retained_ids = set(stage2_plan['retained_samples']['sample_idx'].astype(int))
    removed_ids = set(stage2_plan['removed_samples']['sample_idx'].astype(int))
    all_ids = set(all_samples_df['sample_idx'].astype(int))
    tail_ids = set(tail_df['sample_idx'].astype(int))
    if retained_ids & removed_ids:
        raise ValueError('{} retained/removed Stage 2 partitions overlap'.format(variant))
    if retained_ids | removed_ids != all_ids:
        raise ValueError('{} Stage 2 partitions do not cover the input set'.format(variant))
    if not tail_ids.issubset(retained_ids):
        raise ValueError('{} must retain every TailSampler candidate'.format(variant))


def _build_raw_tail_roles(tail_df):
    tail_open_df = tail_df.copy().sort_values('sample_idx', kind='mergesort').reset_index(drop=True)
    tail_attached_df = _empty_tail_affiliated_frame(tail_df)
    return tail_open_df, tail_attached_df, tail_attached_df.copy()


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


def run_training(args, item_list, device, print_fn):
    setup_seed(1)
    variant = str(args.variant)
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

    model, trainable = build_model(device, encoder_name=args.encoder_name)
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
        manifest_path=args.manifest_path,
    )
    if contaminated_paths is None:
        print_fn('tailguard diagnosis: contamination manifest not found, contamination-aware logging will be unavailable.')
    else:
        print_fn('tailguard diagnosis: loaded contamination manifest {}.'.format(manifest_path))
    args.tg_save_tailsampler_analysis = manifest_path is not None
    args.tg_save_tailsampler_analysis_details = manifest_path is not None

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
            manifest_requested_path=args.manifest_path,
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
    train_data_list = base_train_data_list
    train_data = full_train_data
    train_dataloader = full_train_dataloader
    train_eval_dataloader = full_train_eval_dataloader
    memory_train_eval_dataloader = full_train_eval_dataloader
    print_fn('stage1 training scope=all_samples train image number:{} / full {}'.format(
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
    gbps_has_postprocessed = variant == 'core'
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

            if variant != 'core' and gbps_trigger_iter is None and not gbps_has_postprocessed and should_run_check(
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
                summary['manifest_path'] = manifest_path
                summary['manifest_requested_path'] = args.manifest_path
                summary['has_contamination_labels'] = bool(contaminated_paths is not None)
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
                        'gbps_prune_mode': 'adaptive',
                        'gbps_prune_max_ratio': float(args.gbps_prune_max_ratio),
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
                            if variant_uses_reconciliation(variant):
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
                            elif variant == 'dhp_cte':
                                tail_open_df, tail_head_normal_df, tail_head_noise_df = _build_raw_tail_roles(tail_df)
                                stage2_plan = build_tailguard_stage2_plan(
                                    head_delete_plan['h_clean_samples'],
                                    head_delete_plan['h_removed_samples'],
                                    tail_open_df,
                                    tail_head_normal_df,
                                    tail_head_noise_df,
                                    all_samples_df=method_metadata_df,
                                )
                            elif variant == 'dhp':
                                tail_open_df = tail_df.copy()
                                tail_head_normal_df = _empty_tail_affiliated_frame(tail_df)
                                tail_head_noise_df = tail_head_normal_df.copy()
                                stage2_plan = _build_dhp_stage2_plan(
                                    head_delete_plan,
                                    tail_df,
                                    method_metadata_df,
                                    num_classes=len(base_train_data_list),
                                )
                            else:
                                raise ValueError('unexpected cleanup-capable variant: {}'.format(variant))
                            _validate_stage2_partition(stage2_plan, method_metadata_df, tail_df, variant)
                            stage2_plan['summary'].update({
                                'variant': variant,
                                'reconciliation_performed': variant_uses_reconciliation(variant),
                                'enhancement_performed': variant_uses_enhancement(variant),
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
                                manifest_requested_path=args.manifest_path,
                                label_source=prepare_result['saved'].get('tailguard_train_analysis_metadata_csv') if prepare_result['saved'] else None,
                            )
                            denoising_diagnostics_artifacts = save_tailguard_denoising_diagnostics(
                                args.tg_root_dir,
                                denoising_diagnostics_summary,
                                denoising_diagnostics_by_class,
                            )
                            if variant_uses_enhancement(variant):
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
        if variant == 'core':
            cleanup_reason = 'variant_core'
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
            manifest_requested_path=args.manifest_path,
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
        if not variant_uses_enhancement(variant):
            pseudoclass_reason = 'variant_{}_disables_enhancement'.format(variant)
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
    memory_enabled = bool(args.tg_memory_enable) and variant_uses_enhancement(variant)
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
            'checkpoint_path': checkpoint_path,
            'encoder_name': args.encoder_name,
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

    print_fn('Training stage finished. Total time: {:.2f}s ({:.2f} min), {:.4f} s/iter'.format(total_time, total_time / 60.0, time_per_iter))
    if memory_eval_summary is not None:
        print_fn('Reconciled-reference evaluation:')
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
        'variant': variant,
        'resolved_method_config': args.tailguard_resolved_method_config,
        'dataset_provenance': args.dataset_provenance,
        'checkpoint_path': checkpoint_path,
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
    return {
        'checkpoint_path': checkpoint_path,
        'memory_status': memory_status,
        'memory_artifacts': memory_saved,
        'summary': tailguard_summary,
        'summary_path': summary_path,
    }
