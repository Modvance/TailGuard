"""Single source of truth for the canonical TailGuard configuration."""


TAILGUARD_CONFIG_SCHEMA_VERSION = 8
TAILGUARD_METHOD_NAME = 'tailguard'
TAILGUARD_CONFIG_PROFILE = 'final'
TAILGUARD_VARIANTS = ('core', 'dhp', 'dhp_cte', 'dhp_trp', 'full')


# Training and evaluation settings used for every paper result.  These values
# are deliberately not command-line options: changing them defines a different
# experimental protocol rather than a different TailGuard input.
TAILGUARD_TRAINING_CONFIG = {
    'total_iters': 10000,
    'eval_interval': 5000,
    'image_size': 448,
    'crop_size': 392,
    'encoder_name': 'dinov2reg_vit_base_14',
    'lr': 2e-3,
    'final_lr': 2e-4,
    'warmup_iters': 100,
    'weight_decay': 1e-4,
    'grad_clip_norm': 0.1,
    'max_ratio': 0.01,
    'resize_mask': 256,
}


# Canonical parameters used by the complete TailGuard pipeline.
TAILGUARD_FINAL_DEFAULTS = {
    'tailsampler_percentile': 0.15,
    'gbps_check_start_ratio': 0.01,
    'gbps_check_end_ratio': 0.15,
    'gbps_check_interval': 20,
    'gbps_grouping_method': 'kmeans',
    'gbps_auto_k': True,
    'gbps_k_candidates': '6,8,10,12,15,20',
    'gbps_pca_dim': 64,
    'gbps_min_group_size': 8,
    'gbps_tau_bic': 0.0,
    'gbps_tau_sep': 0.5,
    'gbps_tau_conf': 0.1,
    'gbps_pi_min': 0.05,
    'gbps_pi_max': 0.95,
    'gbps_gate_mode': 'hard',
    'gbps_min_noise_evidence': 0.05,
    'gbps_prune_max_ratio': 0.1,
    'gbps_prune_stable_window': 3,
    'gbps_prune_stable_min_observations': 1,
    'gbps_prune_min_active_ratio': 0.5,
    'gbps_min_keep_per_group': 20,
    'gbps_bootstrap_B': 20,
    'gbps_ci_z': 1.96,
    'gbps_improve_eps': 1e-6,
    'gbps_min_checks_before_trigger': 2,
    'gbps_min_checks_after_best': 1,
    'tg_tail_embedding_source': 'encoder_cls',
    'tg_group_embedding_source': 'encoder',
    'tg_save_grouping_embeddings': True,
    'tg_default_adaptive_angle': 5.0,
    'tg_min_clean_group_size': 3,
    'tg_attachment_membership_mode': 'rgd',
    'tg_elbow_min_segment': 3,
    'tg_rgd_split_mode': 'segmented_bic',
    'tg_memory_fusion_lambda': 1.0,
    'tg_memory_topk_ratio': 0.05,
    # The paper configuration routes enhancement by the predicted tail group.
    'tg_memory_route_margin_threshold': float('-inf'),
    # Tail groups remain eligible even when only one training sample is available.
    'tg_memory_min_class_members': 1,
    'tg_mem_chunk_size': 4096,
    'tg_mem_max_patches_per_class': 20000,
}


def tailguard_final_defaults():
    """Return a copy so callers cannot mutate the public default profile."""
    return dict(TAILGUARD_FINAL_DEFAULTS)


def resolve_variant(variant):
    """Validate and normalize one published paper variant."""
    variant = str(variant)
    if variant not in TAILGUARD_VARIANTS:
        raise ValueError(
            'unknown TailGuard variant {!r}; supported variants: {}'.format(
                variant,
                ', '.join(TAILGUARD_VARIANTS),
            )
        )
    return variant


def variant_uses_reconciliation(variant):
    """Return whether a published variant contains tail reconciliation."""
    return resolve_variant(variant) in {'dhp_trp', 'full'}


def variant_uses_enhancement(variant):
    """Return whether a published variant contains conditional enhancement."""
    return resolve_variant(variant) in {'dhp_cte', 'full'}


def tailguard_paper_config(variant='full'):
    """Return the complete fixed protocol for one published paper variant."""
    config = dict(TAILGUARD_TRAINING_CONFIG)
    config.update(TAILGUARD_FINAL_DEFAULTS)
    variant = resolve_variant(variant)
    enhancement_enabled = variant_uses_enhancement(variant)
    config.update({
        'tg_memory_enable': enhancement_enabled,
        'tg_save_cls_embeddings': enhancement_enabled,
        'gbps_postprocess_mode': 'none' if variant == 'core' else 'remove',
        'gbps_prune_max_ratio': (
            0.0 if variant == 'core'
            else TAILGUARD_FINAL_DEFAULTS['gbps_prune_max_ratio']
        ),
    })
    return config
