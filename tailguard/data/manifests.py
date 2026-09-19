"""Resolve optional contamination manifests used only for post-hoc auditing."""

import os
import re
from pathlib import Path


def _append_candidate(candidates, seen, candidate):
    candidate = Path(candidate)
    candidate_key = str(candidate)
    if candidate_key not in seen:
        seen.add(candidate_key)
        candidates.append(candidate)


def _extract_setting_and_seed(name):
    direct_match = re.match(r"^(step_k\d+|pareto)[-_](seed\d+)$", name)
    if direct_match is not None:
        return direct_match.group(1), direct_match.group(2)

    dataset_match = re.match(r"^(?:mvtecad|visa)-(step_k\d+|pareto)-(seed\d+)$", name)
    if dataset_match is not None:
        return dataset_match.group(1), dataset_match.group(2)

    return None, None


def _resolve_candidates(data_path, repository_root):
    absolute_data_path = os.path.abspath(data_path)
    data_parts = Path(absolute_data_path).parts
    candidates = []
    seen = set()
    normalized_parts = {part.lower() for part in data_parts}
    dataset_name = os.path.basename(os.path.normpath(absolute_data_path)).lower()
    manifest_base = Path(repository_root) / "manifests"
    if dataset_name.startswith("mvtecad-") or "mvtecad" in normalized_parts:
        manifest_roots = (manifest_base / "mvtecad-nlt",)
    elif dataset_name.startswith("visa-") or "visa" in normalized_parts:
        manifest_roots = (manifest_base / "visa-nlt",)
    else:
        manifest_roots = (
            manifest_base / "mvtecad-nlt",
            manifest_base / "visa-nlt",
        )

    for index, part in enumerate(data_parts):
        setting, seed = _extract_setting_and_seed(part)
        if setting is not None and seed is not None:
            for manifest_root in manifest_roots:
                _append_candidate(
                    candidates,
                    seen,
                    manifest_root / setting / seed / "inject_defects.txt",
                )

        if part.startswith("step_k") or part == "pareto":
            if index + 1 < len(data_parts) and data_parts[index + 1].startswith("seed"):
                for manifest_root in manifest_roots:
                    _append_candidate(
                        candidates,
                        seen,
                        manifest_root / part / data_parts[index + 1] / "inject_defects.txt",
                    )

    setting, seed = _extract_setting_and_seed(os.path.basename(os.path.normpath(absolute_data_path)))
    if setting is not None and seed is not None:
        for manifest_root in manifest_roots:
            _append_candidate(
                candidates,
                seen,
                manifest_root / setting / seed / "inject_defects.txt",
            )

    if os.path.isfile(absolute_data_path):
        _append_candidate(candidates, seen, absolute_data_path)

    return candidates


def load_injected_manifest(data_path, repository_root, manifest_path=None):
    """Load injected sample paths for diagnostics without exposing them to the method."""
    candidates = []
    if manifest_path is not None:
        explicit_path = Path(manifest_path).expanduser()
        if not explicit_path.is_file():
            raise FileNotFoundError(
                'explicit manifest does not exist: {}'.format(explicit_path)
            )
        candidates.append(explicit_path)
    else:
        candidates.extend(_resolve_candidates(data_path, repository_root))

    for candidate in candidates:
        if candidate.is_file():
            with candidate.open(encoding="utf-8") as manifest_file:
                paths = {
                    line.strip().replace("\\", "/")
                    for line in manifest_file
                    if line.strip()
                }
            return paths, str(candidate)
    return None, None
