"""Single source of truth for the canonical TailGuard configuration."""


TAILGUARD_CONFIG_SCHEMA_VERSION = 7
TAILGUARD_METHOD_NAME = 'tailguard'
TAILGUARD_CONFIG_PROFILE = 'final'
TAILGUARD_PRUNE_MODES = ('adaptive', 'fixed')
TAILGUARD_METHOD_MODES = ('core', 'h_only', 'h_raw_e', 'full_no_e', 'full')


# Canonical parameters used by the complete TailGuard pipeline.
TAILGUARD_FINAL_DEFAULTS = {
    'tg_method_mode': 'full',
    'tailsampler_type': 'adaptive_trim_mode',
    'tailsampler_th_type': None,
    'tailsampler_vote_type': None,
    'tailsampler_percentile': 0.15,
    'tailsampler_plateau_gap_guard': True,
    'tailsampler_embedding_source': 'encoder_cls',
    'tailsampler_gt_mode': 'dataset_rule',
    'tailsampler_tail_count_thr': 8,
    'tailsampler_dataset_name': None,
    'gbps_check_start_ratio': 0.01,
    'gbps_check_end_ratio': 0.15,
    'gbps_check_interval': 20,
    'gbps_grouping_method': 'kmeans',
    'gbps_num_groups': None,
    'gbps_auto_k': True,
    'gbps_k_candidates': '6,8,10,12,15,20',
    'gbps_pca_dim': 64,
    'gbps_min_group_size': 8,
    'gbps_embedding_source': 'encoder',
    'gbps_tau_bic': 0.0,
    'gbps_tau_sep': 0.5,
    'gbps_tau_conf': 0.1,
    'gbps_pi_min': 0.05,
    'gbps_pi_max': 0.95,
    'gbps_gate_mode': 'hard',
    'gbps_min_noise_evidence': 0.05,
    'gbps_postprocess_mode': 'remove',
    'gbps_prune_mode': 'adaptive',
    'gbps_prune_max_ratio': 0.1,
    'gbps_prune_stable_window': 3,
    'gbps_prune_stable_min_observations': 1,
    'gbps_prune_min_active_ratio': 0.5,
    'gbps_prune_ratio': 0.1,
    'gbps_min_keep_per_group': 20,
    'gbps_bootstrap_B': 20,
    'gbps_ci_z': 1.96,
    'gbps_improve_eps': 1e-6,
    'gbps_min_checks_before_trigger': 2,
    'gbps_min_checks_after_best': 1,
    'tg_tail_embedding_source': 'encoder_cls',
    'tg_group_embedding_source': 'encoder',
    'tg_stage1_training_scope': 'all_samples',
    'tg_save_cls_embeddings': True,
    'tg_save_grouping_embeddings': True,
    'tg_save_tailsampler_analysis': True,
    'tg_save_tailsampler_analysis_details': True,
    'tg_default_adaptive_angle': 5.0,
    'tg_min_clean_group_size': 3,
    'tg_attachment_membership_mode': 'rgd',
    'tg_elbow_min_segment': 3,
    'tg_rgd_split_mode': 'segmented_bic',
    'tg_stable_window': 3,
    'tg_stable_min_observations': 1,
    # TailGuard protects every TailSampler candidate from direct deletion.
    'tg_t_attached_noise_p_high_thr': 1.0,
    'tg_memory_enable': True,
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
