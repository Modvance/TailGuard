"""Dataset profiles and provenance contracts for TailGuard."""

from dataclasses import dataclass
import os
from typing import Dict, Tuple


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    item_list: Tuple[str, ...]
    default_data_path: str
    layout: str = 'mvtec_compatible'


MVTec_PROFILE = DatasetProfile(
    name='mvtec',
    item_list=(
        'carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
        'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
    ),
    default_data_path='../mvtec_anomaly_detection',
)

VISA_PROFILE = DatasetProfile(
    name='visa',
    item_list=(
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
        'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum',
    ),
    default_data_path='../VisA_pytorch/1cls',
)

_DATASET_PROFILES: Dict[str, DatasetProfile] = {
    MVTec_PROFILE.name: MVTec_PROFILE,
    VISA_PROFILE.name: VISA_PROFILE,
}


def dataset_profile_names() -> Tuple[str, ...]:
    return tuple(_DATASET_PROFILES)


def get_dataset_profile(name: str) -> DatasetProfile:
    profile_name = str(name).strip().lower()
    try:
        return _DATASET_PROFILES[profile_name]
    except KeyError as error:
        raise ValueError(
            'unknown TailGuard dataset profile {!r}; supported profiles: {}'.format(
                name,
                ', '.join(dataset_profile_names()),
            )
        ) from error


def profile_provenance(profile: DatasetProfile, data_path: str) -> Dict[str, object]:
    return {
        'profile_name': profile.name,
        'item_list': list(profile.item_list),
        'layout': profile.layout,
        'data_path': os.path.abspath(data_path),
    }


def validate_profile_contract(profile: DatasetProfile, profile_name, item_list, source: str):
    if profile_name is not None:
        saved_profile = get_dataset_profile(profile_name)
        if saved_profile.name != profile.name:
            raise ValueError(
                '{} profile {} does not match active profile {}'.format(
                    source,
                    saved_profile.name,
                    profile.name,
                )
            )
    if item_list is None:
        return
    if isinstance(item_list, str):
        raise ValueError('{} item_list must be an ordered sequence, not a string'.format(source))
    try:
        saved_item_list = tuple(str(item) for item in item_list)
    except TypeError as error:
        raise ValueError('{} item_list must be an ordered sequence'.format(source)) from error
    if saved_item_list != profile.item_list:
        raise ValueError('{} item_list does not match {} profile order'.format(source, profile.name))
