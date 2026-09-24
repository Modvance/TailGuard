"""Final coverage-reference evaluation and score fusion for TailGuard."""

import json
import os
import time
from types import SimpleNamespace

from tailguard.method.coverage import replay_coverage
from tailguard.method.score_fusion import fuse as fuse_dual_scores
from tailguard.reporting.compact_run import compact_completed_run


def restore_completed_training(args):
    """Restore the saved training result after a final-evaluation interruption."""
    run_dir = os.path.realpath(os.path.join(args.save_dir, args.save_name))
    checkpoint_path = os.path.join(run_dir, 'final_model.pt')
    summary_path = os.path.join(args.tg_work_dir, 'tailguard_summary.json')
    memory_scores_path = os.path.join(
        args.tg_work_dir,
        'memory',
        'memory_eval_scores.csv',
    )
    required = (checkpoint_path, summary_path, memory_scores_path)
    existing = [os.path.isfile(path) for path in required]
    if not any(existing):
        return None
    if not all(existing):
        missing = [path for path, present in zip(required, existing) if not present]
        raise RuntimeError(
            'the run directory contains an incomplete training result; missing: {}'.format(
                ', '.join(missing)
            )
        )

    with open(summary_path, encoding='utf-8') as summary_file:
        summary = json.load(summary_file)
    if summary.get('variant') != 'full':
        raise RuntimeError(
            'saved training result uses variant {!r}, not full'.format(
                summary.get('variant')
            )
        )
    if summary.get('memory_status') != 'completed':
        raise RuntimeError('saved training result has no completed memory evaluation')

    provenance = summary.get('dataset_provenance') or {}
    saved_profile = provenance.get('profile_name')
    if saved_profile is not None and saved_profile != args.dataset_profile:
        raise RuntimeError(
            'saved dataset profile {} does not match {}'.format(
                saved_profile,
                args.dataset_profile,
            )
        )
    saved_data_path = provenance.get('data_path')
    if (
        saved_data_path is not None
        and os.path.realpath(saved_data_path) != os.path.realpath(args.data_path)
    ):
        raise RuntimeError(
            'saved data path does not match the requested data path: {} != {}'.format(
                saved_data_path,
                os.path.realpath(args.data_path),
            )
        )

    memory_artifacts = dict(summary.get('memory_artifacts') or {})
    memory_artifacts['memory_eval_scores_csv'] = memory_scores_path
    return {
        'checkpoint_path': checkpoint_path,
        'memory_status': 'completed',
        'memory_artifacts': memory_artifacts,
        'summary': summary,
        'summary_path': summary_path,
        'restored': True,
    }


def run_final_evaluation(args, profile, training_result, print_fn):
    """Finish the canonical full method after training and memory evaluation."""
    evaluation_start_time = time.time()
    if training_result['memory_status'] != 'completed':
        raise RuntimeError(
            'the full TailGuard pipeline cannot finish because reconciled '
            'memory evaluation was not completed'
        )

    run_dir = os.path.realpath(os.path.join(args.save_dir, args.save_name))
    coverage_dir = os.path.join(args.tg_work_dir, 'coverage')
    final_dir = os.path.join(args.tg_work_dir, 'final')
    coverage_summary_path = os.path.join(coverage_dir, 'coverage_replay_summary.json')
    coverage_scores = os.path.join(coverage_dir, 'coverage_eval_scores.csv')
    coverage_memory = os.path.join(coverage_dir, 'coverage_memory_system.pt')
    coverage_complete = all(os.path.isfile(path) for path in (
        coverage_summary_path,
        coverage_scores,
        coverage_memory,
    ))
    if os.path.exists(coverage_dir) and not coverage_complete:
        raise RuntimeError('incomplete coverage output requires manual inspection: {}'.format(coverage_dir))
    if coverage_complete:
        print_fn('Restored completed coverage evaluation: {}'.format(coverage_dir))
        with open(coverage_summary_path, encoding='utf-8') as summary_file:
            coverage_summary = json.load(summary_file)
    else:
        print_fn('Building TailGuard coverage reference from the trained checkpoint...')
        feature_cache_dir = os.path.join(args.tg_work_dir, 'feature_cache')
        coverage_summary = replay_coverage(SimpleNamespace(
            full_run_dir=run_dir,
            data_path=args.data_path,
            output_dir=coverage_dir,
            feature_cache_dir=feature_cache_dir,
            encoder_checkpoint_path=training_result['checkpoint_path'],
            dataset_profile=profile.name,
            reference_raw_run_dir=None,
            gpu=int(args.gpus),
            batch_size=min(4, int(args.batch_size)),
            tg_memory_topk_ratio=float(args.tg_memory_topk_ratio),
            tg_memory_fusion_lambda=float(args.tg_memory_fusion_lambda),
            tg_memory_route_margin_threshold=float(
                args.tg_memory_route_margin_threshold
            ),
            tg_memory_min_class_members=int(args.tg_memory_min_class_members),
            float_atol=2e-5,
            no_save_memory_system=False,
        ))

    analysis_artifacts = (
        training_result['summary']
        .get('prepare_summary', {})
        .get('tail_sampler_analysis_only_artifacts')
    )
    class_roles_path = None
    if isinstance(analysis_artifacts, dict):
        candidate = analysis_artifacts.get('sampler_analysis_details_csv')
        if candidate and os.path.isfile(candidate):
            class_roles_path = candidate

    reconciled_scores = (
        training_result['memory_artifacts']['memory_eval_scores_csv']
    )
    final_summary_path = os.path.join(final_dir, 'dual_summary.json')
    final_scores_path = os.path.join(final_dir, 'dual_scores.csv')
    final_per_class_path = os.path.join(final_dir, 'dual_per_class.csv')
    final_complete = all(os.path.isfile(path) for path in (
        final_summary_path,
        final_scores_path,
        final_per_class_path,
    ))
    if os.path.exists(final_dir) and not final_complete:
        raise RuntimeError('incomplete final fusion output requires manual inspection: {}'.format(final_dir))
    if final_complete:
        print_fn('Restored completed score fusion: {}'.format(final_dir))
        with open(final_summary_path, encoding='utf-8') as summary_file:
            final_image_summary = json.load(summary_file)
    else:
        print_fn(
            'Fusing coverage and reconciled evidence into the final TailGuard output...'
        )
        final_image_summary = fuse_dual_scores(
            coverage_path=coverage_scores,
            reconciled_path=reconciled_scores,
            output_dir=final_dir,
            class_roles_path=class_roles_path,
        )

    complete_summary = dict(training_result['summary'])
    final_evaluation_time = time.time() - evaluation_start_time
    complete_summary['final_detection'] = {
        'status': 'completed',
        'image_score_protocol': 'mean of coverage and reconciled image scores',
        'image_metrics': final_image_summary,
        'pixel_metrics': training_result['summary'].get('memory_eval_summary'),
        'coverage_summary': coverage_summary,
        'evaluation_time_s': float(final_evaluation_time),
        'artifacts': {
            'coverage_scores_csv': coverage_scores,
            'final_scores_csv': final_scores_path,
            'final_per_class_csv': final_per_class_path,
            'final_summary_json': final_summary_path,
        },
    }
    complete_summary['end_to_end_time_s'] = float(
        complete_summary.get('total_time_s', 0.0) + final_evaluation_time
    )
    compact_summary = compact_completed_run(args, complete_summary)
    summary_path = os.path.join(args.tg_root_dir, 'run_summary.json')
    print_fn('Complete TailGuard evaluation finished.')
    print_fn(
        '  final I-AUROC  {:.2f}'.format(
            100.0 * final_image_summary['all_I-AUROC']
        )
    )
    print_fn('  evaluation time {:.2f}s'.format(final_evaluation_time))
    print_fn('  summary        {}'.format(summary_path))
    return compact_summary
