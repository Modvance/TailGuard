"""Final coverage-reference evaluation and score fusion for TailGuard."""

import os
import time
from types import SimpleNamespace

from tailguard.method.coverage import replay_coverage
from tailguard.method.score_fusion import fuse as fuse_dual_scores
from tailguard.reporting.artifacts import save_tailguard_summary


def run_final_evaluation(args, profile, training_result, print_fn):
    """Finish the canonical full method after training and memory evaluation."""
    evaluation_start_time = time.time()
    if training_result['memory_status'] != 'completed':
        raise RuntimeError(
            'the full TailGuard pipeline cannot finish because reconciled '
            'memory evaluation was not completed'
        )

    run_dir = os.path.realpath(os.path.join(args.save_dir, args.save_name))
    coverage_dir = os.path.join(args.tg_root_dir, 'coverage')
    final_dir = os.path.join(args.tg_root_dir, 'final')
    feature_cache_dir = os.path.join(args.tg_root_dir, 'coverage_feature_cache')
    for output_dir in (coverage_dir, final_dir):
        if os.path.exists(output_dir):
            raise FileExistsError(
                'final evaluation output already exists; use a new '
                '--save_name: {}'.format(output_dir)
            )

    print_fn('Building TailGuard coverage reference from the trained checkpoint...')
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
    coverage_scores = os.path.join(
        coverage_dir,
        'coverage_eval_scores.csv',
    )
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
            'final_scores_csv': os.path.join(final_dir, 'dual_scores.csv'),
            'final_per_class_csv': os.path.join(final_dir, 'dual_per_class.csv'),
            'final_summary_json': os.path.join(final_dir, 'dual_summary.json'),
        },
    }
    complete_summary['end_to_end_time_s'] = float(
        complete_summary.get('total_time_s', 0.0) + final_evaluation_time
    )
    summary_path = save_tailguard_summary(args.tg_root_dir, complete_summary)
    print_fn('Complete TailGuard evaluation finished.')
    print_fn(
        '  final I-AUROC  {:.2f}'.format(
            100.0 * final_image_summary['all_I-AUROC']
        )
    )
    print_fn('  evaluation time {:.2f}s'.format(final_evaluation_time))
    print_fn('  summary        {}'.format(summary_path))
    return complete_summary
