import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


DEFAULT_K_CANDIDATES = (8, 10, 12, 15, 20)


def _parse_k_candidates(values):
    if values is None:
        return DEFAULT_K_CANDIDATES
    if isinstance(values, str):
        return tuple(int(part.strip()) for part in values.split(',') if part.strip())
    return tuple(int(value) for value in values)


def _to_python_scalar(value):
    if torch.is_tensor(value):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


def _extract_encoder_gap_embeddings(model, images: torch.Tensor) -> np.ndarray:
    en, _ = model(images)
    feature_map = en[-1]
    pooled = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)
    return pooled.detach().cpu().numpy().astype(np.float32)


def _extract_encoder_cls_embeddings(model, images: torch.Tensor) -> np.ndarray:
    encoder = getattr(model, 'encoder', None)
    if encoder is None:
        raise ValueError('model does not expose encoder for cls embedding extraction')
    if not hasattr(encoder, 'prepare_tokens_with_masks'):
        raise ValueError('encoder_cls embedding_source requires a DINOv2-style encoder')

    x = encoder.prepare_tokens_with_masks(images)
    for blk in encoder.blocks:
        x = blk(x)
    x = encoder.norm(x)
    cls_token = x[:, 0]
    return cls_token.detach().cpu().numpy().astype(np.float32)


def extract_image_embeddings(model, dataloader, device, embedding_source='encoder'):
    if embedding_source not in {'encoder', 'encoder_cls'}:
        raise ValueError(f"unsupported embedding_source: {embedding_source}")

    was_training = model.training
    rows = []
    embedding_list = []

    model.eval()
    with torch.no_grad():
        for images, _, meta in dataloader:
            images = images.to(device)
            if embedding_source == 'encoder':
                batch_embeddings = _extract_encoder_gap_embeddings(model, images)
            else:
                batch_embeddings = _extract_encoder_cls_embeddings(model, images)

            batch_size = batch_embeddings.shape[0]
            for index in range(batch_size):
                row = {}
                for key, value in meta.items():
                    row[key] = _to_python_scalar(value[index])
                row['sample_key'] = int(row['sample_idx'])
                rows.append(row)
                embedding_list.append(batch_embeddings[index])

    if was_training:
        model.train()

    metadata = pd.DataFrame(rows)
    embeddings = np.stack(embedding_list, axis=0) if len(embedding_list) > 0 else np.zeros((0, 0), dtype=np.float32)
    return embeddings, metadata


def fit_latent_groups(
    embeddings,
    sample_keys,
    method='kmeans',
    num_groups=None,
    auto_k=True,
    k_candidates=DEFAULT_K_CANDIDATES,
    pca_dim=64,
    random_state=0,
):
    if method != 'kmeans':
        raise ValueError(f"unsupported grouping method: {method}")

    sample_keys = [int(key) for key in sample_keys]
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError('embeddings must be a 2D array')
    if embeddings.shape[0] != len(sample_keys):
        raise ValueError('embeddings and sample_keys length mismatch')
    if embeddings.shape[0] == 0:
        raise ValueError('cannot fit latent groups on empty embeddings')

    normalized = _normalize_embeddings(embeddings)
    selected_pca_dim = None
    if pca_dim is not None and pca_dim > 0 and normalized.shape[1] > pca_dim and normalized.shape[0] > pca_dim:
        selected_pca_dim = int(pca_dim)
        normalized = PCA(n_components=selected_pca_dim, random_state=random_state).fit_transform(normalized)

    if num_groups is not None:
        selected_k = int(num_groups)
        best_score = None
    else:
        candidates = _parse_k_candidates(k_candidates)
        valid_candidates = [k for k in candidates if 2 <= int(k) < len(sample_keys)]
        if len(valid_candidates) == 0:
            selected_k = max(1, min(len(sample_keys), 2))
            best_score = None
        elif not auto_k:
            selected_k = int(valid_candidates[0])
            best_score = None
        else:
            best_score = None
            selected_k = int(valid_candidates[0])
            for candidate in valid_candidates:
                kmeans = KMeans(n_clusters=int(candidate), random_state=random_state, n_init=10)
                labels = kmeans.fit_predict(normalized)
                if len(np.unique(labels)) < 2:
                    continue
                score = silhouette_score(normalized, labels)
                if best_score is None or score > best_score:
                    best_score = float(score)
                    selected_k = int(candidate)

    if selected_k <= 1:
        assignments = np.zeros(len(sample_keys), dtype=int)
    else:
        kmeans = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
        assignments = kmeans.fit_predict(normalized)

    group_assignments = {
        int(sample_key): int(group_id)
        for sample_key, group_id in zip(sample_keys, assignments)
    }
    group_sizes = pd.Series(assignments).value_counts().sort_index().to_dict()
    group_info = {
        'method': method,
        'selected_k': int(selected_k),
        'auto_k': bool(auto_k and num_groups is None),
        'silhouette_score': None if best_score is None else float(best_score),
        'pca_dim': None if selected_pca_dim is None else int(selected_pca_dim),
        'group_sizes': {str(int(group_id)): int(size) for group_id, size in group_sizes.items()},
        'num_samples': int(len(sample_keys)),
    }
    return group_assignments, group_info

