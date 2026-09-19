"""Train and evaluate TailGuard through one public entry point."""

import argparse
import os

import torch

from tailguard.config import (
    TAILGUARD_CONFIG_PROFILE,
    TAILGUARD_CONFIG_SCHEMA_VERSION,
    TAILGUARD_FINAL_DEFAULTS,
    TAILGUARD_METHOD_NAME,
    TAILGUARD_VARIANTS,
    tailguard_paper_config,
    variant_uses_enhancement,
    variant_uses_reconciliation,
)
from tailguard.data.profiles import (
    dataset_profile_names,
    get_dataset_profile,
    profile_provenance,
)
from tailguard.engine.final_evaluation import run_final_evaluation
from tailguard.engine.pipeline import run_training
from tailguard.engine.trainer import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    get_logger,
)


def build_parser(default_dataset_profile='mvtec'):
    """Build the small public interface for a complete TailGuard run."""
    default_profile = get_dataset_profile(default_dataset_profile)
    parser = argparse.ArgumentParser(
        description='Train and evaluate TailGuard on an MVTec-compatible dataset'
    )
    parser.add_argument(
        '--dataset_profile',
        choices=dataset_profile_names(),
        default=default_profile.name,
        help='Dataset profile; also supplies the default data path.',
    )
    parser.add_argument(
        '--data_path',
        default=None,
        help='Dataset root. Defaults to the path defined by the selected profile.',
    )
    parser.add_argument(
        '--save_dir',
        default='./saved_results',
        help='Directory in which run folders are created.',
    )
    parser.add_argument(
        '--save_name',
        default=None,
        help='Run folder name. A profile-based name is used when omitted.',
    )
    parser.add_argument(
        '--manifest_path',
        default=None,
        help=(
            'Optional injected-anomaly manifest for post-hoc auditing. '
            'Automatically resolved for datasets built by build_datasets.sh.'
        ),
    )
    parser.add_argument(
        '--gpu',
        dest='gpus',
        type=int,
        default=0,
        help='CUDA device index.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Training batch size; the paper configuration uses 16.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help='Data-loader worker count.',
    )
    parser.add_argument(
        '--variant',
        default='full',
        choices=TAILGUARD_VARIANTS,
        help='Paper configuration to reproduce; full is the canonical method.',
    )
    parser.add_argument(
        '--memory_chunk_size',
        dest='tg_mem_chunk_size',
        type=int,
        default=TAILGUARD_FINAL_DEFAULTS['tg_mem_chunk_size'],
        help='Memory-distance chunk size; reduce only when GPU memory is limited.',
    )
    return parser


def _apply_paper_configuration(args):
    for name, value in tailguard_paper_config(args.variant).items():
        if name == 'tg_mem_chunk_size' and hasattr(args, name):
            continue
        setattr(args, name, value)

    args.diag_save_dir = 'gbps'
    args.diag_batch_size = args.batch_size
    args.diag_num_workers = args.num_workers
    args.diag_max_ratio = args.max_ratio
    args.diag_resize_mask = args.resize_mask


def _resolve_dataset_profile(args):
    profile = get_dataset_profile(args.dataset_profile)
    if args.data_path is None:
        args.data_path = profile.default_data_path
    if args.save_name is None:
        args.save_name = 'vitill_{}_tailguard'.format(profile.name)
    args.dataset_profile = profile.name
    args.dataset_item_list = list(profile.item_list)
    args.dataset_layout = profile.layout
    args.dataset_provenance = profile_provenance(profile, args.data_path)
    return profile


def _resolved_method_config(args):
    config = {
        name: getattr(args, name, default)
        for name, default in TAILGUARD_FINAL_DEFAULTS.items()
    }
    config.update({
        'variant': args.variant,
        'reconciliation_enabled': variant_uses_reconciliation(args.variant),
        'enhancement_enabled': variant_uses_enhancement(args.variant),
        'head_postprocess_enabled': args.gbps_postprocess_mode == 'remove',
    })
    return config


def _prepare_run_directories(args):
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


def main(argv=None):
    """Run training, inference, and evaluation as one TailGuard command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_paper_configuration(args)
    profile = _resolve_dataset_profile(args)
    if not os.path.isdir(args.data_path):
        parser.error('dataset directory does not exist: {}'.format(args.data_path))

    expected_memory = variant_uses_enhancement(args.variant)
    if bool(args.tg_memory_enable) != expected_memory:
        raise RuntimeError('resolved variant has inconsistent memory configuration')

    args.tailguard_config_schema_version = TAILGUARD_CONFIG_SCHEMA_VERSION
    args.tailguard_method_name = TAILGUARD_METHOD_NAME
    args.tailguard_config_profile = TAILGUARD_CONFIG_PROFILE
    args.tailguard_resolved_method_config = _resolved_method_config(args)
    _prepare_run_directories(args)

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:{}'.format(args.gpus) if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    training_result = run_training(
        args,
        profile.item_list,
        device,
        print_fn,
    )
    if args.variant == 'full':
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_final_evaluation(args, profile, training_result, print_fn)
    return args


if __name__ == '__main__':
    main()
