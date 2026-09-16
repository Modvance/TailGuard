"""Training and evaluation foundations used by TailGuard."""

import logging
import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder

# Compatibility for the legacy SciPy/scikit-learn versions used by the paper
# environment when NumPy has been upgraded independently.
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "typeDict"):
    np.typeDict = np.sctypeDict

from tailguard.data.datasets import MVTecDataset, TrainDiagDataset, get_data_transforms
from tailguard.models import encoder as vit_encoder
from tailguard.models.reconstruction import ViTill
from tailguard.models.decoder import Block as VitBlock, LinearAttention2, bMlp
from tailguard.optimizers import StableAdamW
from torch.nn.init import trunc_normal_
from tailguard.engine.ops import (
    WarmCosineScheduler,
    compute_image_level_scores,
    evaluation_batch,
    get_gaussian_kernel,
    global_cosine_hm_percent,
    infer_anomaly_map_batch,
)


DEFAULT_TOTAL_ITERS = 10000
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 4
DEFAULT_IMAGE_SIZE = 448
DEFAULT_CROP_SIZE = 392
DEFAULT_ENCODER_NAME = "dinov2reg_vit_base_14"
DEFAULT_TARGET_LAYERS = [2, 3, 4, 5, 6, 7, 8, 9]
DEFAULT_FUSE_LAYER_ENCODER = [[0, 1, 2, 3], [4, 5, 6, 7]]
DEFAULT_FUSE_LAYER_DECODER = [[0, 1, 2, 3], [4, 5, 6, 7]]


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    logger.handlers = []

    log_format = logging.Formatter('%(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_train_dataloader(train_datasets, batch_size, num_workers, sampler=None, shuffle=None, epoch_num_samples=None):
    train_data = ConcatDataset(train_datasets)
    if sampler is None:
        effective_shuffle = True if shuffle is None else bool(shuffle)
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=effective_shuffle,
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        if shuffle is None:
            shuffle = False
        if bool(shuffle):
            raise ValueError('shuffle must be False when sampler is provided')
        if epoch_num_samples is None:
            epoch_num_samples = len(train_data)
        weighted_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sampler, dtype=torch.double),
            num_samples=int(epoch_num_samples),
            replacement=True,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=weighted_sampler,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
        )
    return train_data, train_dataloader


def build_train_eval_dataloader(train_datasets, data_root, batch_size, num_workers, item_list, contaminated_paths=None):
    train_eval_data_list = []
    sample_offset = 0
    for class_id, (item, train_data) in enumerate(zip(item_list, train_datasets)):
        train_eval_data = TrainDiagDataset(
            train_data,
            data_root=data_root,
            class_name=item,
            class_id=class_id,
            sample_offset=sample_offset,
            contaminated_paths=contaminated_paths,
        )
        train_eval_data_list.append(train_eval_data)
        sample_offset += len(train_eval_data)

    train_eval_data = ConcatDataset(train_eval_data_list)
    train_eval_dataloader = torch.utils.data.DataLoader(
        train_eval_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_eval_dataloader


def rebuild_pruned_train_loaders(train_data_list, retained_index_map, data_root, item_list,
                                 batch_size, num_workers, diag_batch_size, diag_num_workers,
                                 contaminated_paths=None, sampler=None, epoch_num_samples=None):
    pruned_train_data_list = []
    for class_id, train_dataset in enumerate(train_data_list):
        retained_indices = retained_index_map.get(class_id)
        if retained_indices is None:
            retained_indices = list(range(len(train_dataset)))
        pruned_train_data_list.append(Subset(train_dataset, retained_indices))

    train_data, train_dataloader = build_train_dataloader(
        pruned_train_data_list,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        epoch_num_samples=epoch_num_samples,
    )
    train_eval_dataloader = build_train_eval_dataloader(
        pruned_train_data_list,
        data_root=data_root,
        batch_size=diag_batch_size,
        num_workers=diag_num_workers,
        item_list=item_list,
        contaminated_paths=contaminated_paths,
    )
    return {
        'train_data_list': pruned_train_data_list,
        'train_data': train_data,
        'train_dataloader': train_dataloader,
        'train_eval_dataloader': train_eval_dataloader,
    }


def build_datasets(data_path, item_list, image_size=DEFAULT_IMAGE_SIZE, crop_size=DEFAULT_CROP_SIZE):
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)
    train_data_list = []
    test_data_list = []

    for class_id, item in enumerate(item_list):
        train_path = os.path.join(data_path, item, 'train')
        test_path = os.path.join(data_path, item)

        train_data = ImageFolder(root=train_path, transform=data_transform)
        train_data.classes = item
        train_data.class_to_idx = {item: class_id}
        train_data.samples = [(sample[0], class_id) for sample in train_data.samples]

        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase='test')
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    return train_data_list, test_data_list


def build_model(device, encoder_name=DEFAULT_ENCODER_NAME, target_layers=None,
                fuse_layer_encoder=None, fuse_layer_decoder=None,
                load_encoder_weights=True):
    target_layers = DEFAULT_TARGET_LAYERS if target_layers is None else list(target_layers)
    fuse_layer_encoder = DEFAULT_FUSE_LAYER_ENCODER if fuse_layer_encoder is None else fuse_layer_encoder
    fuse_layer_decoder = DEFAULT_FUSE_LAYER_DECODER if fuse_layer_decoder is None else fuse_layer_decoder

    encoder = vit_encoder.load(encoder_name, pretrained=load_encoder_weights)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise RuntimeError('Architecture not in small, base, large.')

    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])

    decoder = []
    for _ in range(8):
        blk = VitBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=LinearAttention2,
        )
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
    ).to(device)

    trainable = nn.ModuleList([bottleneck, decoder])
    for module in trainable.modules():
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    return model, trainable


def build_optimizer_and_scheduler(trainable, total_iters=DEFAULT_TOTAL_ITERS, lr=2e-3,
                                  final_lr=2e-4, warmup_iters=100, weight_decay=1e-4):
    optimizer = StableAdamW(
        [{'params': trainable.parameters()}],
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
        amsgrad=True,
        eps=1e-10,
    )
    scheduler = WarmCosineScheduler(
        optimizer,
        base_value=lr,
        final_value=final_lr,
        total_iters=total_iters,
        warmup_iters=warmup_iters,
    )
    return optimizer, scheduler


def train_step(model, trainable, optimizer, scheduler, images, current_zero_based_iter,
               sample_weight=None, grad_clip_norm=0.1):
    model.train()
    images = images.to(next(model.parameters()).device)
    en, de = model(images)

    p_final = 0.9
    p = min(p_final * current_zero_based_iter / 1000, p_final)
    loss = global_cosine_hm_percent(en, de, p=p, factor=0.1, sample_weight=sample_weight)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=grad_clip_norm)
    optimizer.step()
    scheduler.step()
    return float(loss.item())


def save_train_checkpoint(save_dir, save_name, model, iteration, args, final_eval_summary=None):
    checkpoint_dir = os.path.join(save_dir, save_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pt')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'iteration': int(iteration),
        'args': vars(args).copy(),
    }
    if final_eval_summary is not None:
        checkpoint['final_eval_summary'] = final_eval_summary
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def _checkpoint_args_get(checkpoint_args, key, default=None):
    if checkpoint_args is None:
        return default
    if isinstance(checkpoint_args, dict):
        return checkpoint_args.get(key, default)
    return getattr(checkpoint_args, key, default)


def load_train_checkpoint_metadata(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_args = checkpoint.get('args') or {}
    encoder_name = _checkpoint_args_get(checkpoint_args, 'encoder_name', DEFAULT_ENCODER_NAME)
    image_size = int(_checkpoint_args_get(checkpoint_args, 'image_size', DEFAULT_IMAGE_SIZE))
    crop_size = int(_checkpoint_args_get(checkpoint_args, 'crop_size', DEFAULT_CROP_SIZE))
    target_layers = list(_checkpoint_args_get(checkpoint_args, 'target_layers', DEFAULT_TARGET_LAYERS))
    fuse_layer_encoder = _checkpoint_args_get(checkpoint_args, 'fuse_layer_encoder', DEFAULT_FUSE_LAYER_ENCODER)
    fuse_layer_decoder = _checkpoint_args_get(checkpoint_args, 'fuse_layer_decoder', DEFAULT_FUSE_LAYER_DECODER)
    diag_max_ratio = float(_checkpoint_args_get(checkpoint_args, 'diag_max_ratio', 0.01))
    diag_resize_mask = int(_checkpoint_args_get(checkpoint_args, 'diag_resize_mask', 256))
    return {
        'checkpoint_path': checkpoint_path,
        'checkpoint': checkpoint,
        'model_state_dict': checkpoint['model_state_dict'],
        'iteration': checkpoint.get('iteration'),
        'args': checkpoint_args,
        'encoder_name': encoder_name,
        'image_size': image_size,
        'crop_size': crop_size,
        'target_layers': target_layers,
        'fuse_layer_encoder': fuse_layer_encoder,
        'fuse_layer_decoder': fuse_layer_decoder,
        'diag_max_ratio': diag_max_ratio,
        'diag_resize_mask': diag_resize_mask,
        'final_eval_summary': checkpoint.get('final_eval_summary'),
    }


def load_model_from_train_checkpoint(checkpoint_path, device):
    metadata = load_train_checkpoint_metadata(checkpoint_path)
    model, trainable = build_model(
        device,
        encoder_name=metadata['encoder_name'],
        target_layers=metadata['target_layers'],
        fuse_layer_encoder=metadata['fuse_layer_encoder'],
        fuse_layer_decoder=metadata['fuse_layer_decoder'],
        load_encoder_weights=False,
    )
    model.load_state_dict(metadata['model_state_dict'])
    model.eval()
    return model, trainable, metadata


def evaluate_model(model, test_data_list, item_list, device, batch_size=DEFAULT_BATCH_SIZE,
                   num_workers=DEFAULT_NUM_WORKERS, max_ratio=0.01, resize_mask=256, print_fn=None):
    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

    for item, test_data in zip(item_list, test_data_list):
        test_dataloader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        results = evaluation_batch(model, test_dataloader, device, max_ratio=max_ratio, resize_mask=resize_mask)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

        auroc_sp_list.append(auroc_sp)
        ap_sp_list.append(ap_sp)
        f1_sp_list.append(f1_sp)
        auroc_px_list.append(auroc_px)
        ap_px_list.append(ap_px)
        f1_px_list.append(f1_px)
        aupro_px_list.append(aupro_px)

        if print_fn is not None:
            print_fn(
                '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                    item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px,
                )
            )

    summary = {
        'I-AUROC': float(np.mean(auroc_sp_list)),
        'I-AP': float(np.mean(ap_sp_list)),
        'I-F1': float(np.mean(f1_sp_list)),
        'P-AUROC': float(np.mean(auroc_px_list)),
        'P-AP': float(np.mean(ap_px_list)),
        'P-F1': float(np.mean(f1_px_list)),
        'P-AUPRO': float(np.mean(aupro_px_list)),
    }

    if print_fn is not None:
        print_fn(
            'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                summary['I-AUROC'], summary['I-AP'], summary['I-F1'], summary['P-AUROC'], summary['P-AP'], summary['P-F1'], summary['P-AUPRO'],
            )
        )
    return summary


def score_trainset(model, loader, device, max_ratio=0.01, resize_mask=256):
    rows = []
    was_training = model.training
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    model.eval()
    with torch.no_grad():
        for images, _, meta in loader:
            images = images.to(device)
            anomaly_map, _ = infer_anomaly_map_batch(model, images, gaussian_kernel, resize_mask=resize_mask)
            image_scores = compute_image_level_scores(anomaly_map, max_ratio=max_ratio).detach().cpu().numpy()

            batch_size = len(image_scores)
            for index in range(batch_size):
                row = {}
                for key, value in meta.items():
                    item = value[index]
                    if torch.is_tensor(item):
                        item = item.item()
                    elif isinstance(item, np.generic):
                        item = item.item()
                    row[key] = item
                row['image_score'] = float(image_scores[index])
                rows.append(row)

    if was_training:
        model.train()

    import pandas as pd
    return pd.DataFrame(rows)


def should_run_check(current_iter, total_iters, start_ratio, end_ratio, interval):
    t_min = max(1, int(total_iters * start_ratio))
    t_max = max(t_min, int(total_iters * end_ratio))
    if current_iter < t_min or current_iter > t_max:
        return False
    return (current_iter - t_min) % interval == 0 or current_iter == t_max
