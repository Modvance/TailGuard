#!/usr/bin/env python3
"""
make_longtail_noisy_mvtecad.py
==============================
Produce a **long‑tail noisy** variant of the MVTec‑AD dataset using **two
manifest files**:

* **`--noisy-manifest`** — relative paths of *defect* images that should be
  injected into each object class' `train/good` folder.
* **`--prune-manifest`** — relative paths of *normal* (original good) images
  that should be **removed** from `train/good` to yield an imbalanced, long‑tail
  distribution.

Inputs
------
```
--source-dir      pristine MVTec‑AD root (contains full `test` & `ground_truth`)
--dest-dir        destination root where the long‑tail noisy dataset is built
--noisy-manifest  text file listing defect images to inject (relative to dest)
--prune-manifest  text file listing good images to delete (relative to dest)
--symlink         (optional) use symbolic links for injected images instead of copying
--symlink-all     (optional) link every image and safely convert identical existing copies
```

Behaviour
---------
1. Ensures `train/good`, `test`, and `ground_truth` under *dest* mirror those
   in *source* (they are copied only if missing).
2. **Deletes** every path listed in *prune‑manifest* from the destination.
3. **Injects** every path in *noisy‑manifest* by copying/symlinking the
   corresponding file from
   `source/<object>/test/<defect>/<filename>` to
   `dest/<object>/train/good/<defect>_<filename>`.

Both manifests must list paths **relative to the destination root**, matching
what `make_noisy_mvtecad.py` produced (e.g.
`bottle/train/good/broken_small_001.png`).
"""
import argparse
import filecmp
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Dict, List, Union

###############################################################################
# Helpers
###############################################################################

def ensure_writable(path: Path):
    mode = path.stat().st_mode
    if path.is_dir():
        path.chmod(mode | stat.S_IWUSR | stat.S_IXUSR)
    else:
        path.chmod(mode | stat.S_IWUSR)


def ensure_tree_writable(path: Path):
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        ensure_writable(path)
        return
    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        ensure_writable(root_path)
        for name in dirs:
            ensure_writable(root_path / name)
        for name in files:
            ensure_writable(root_path / name)


def _replace_identical_file_with_symlink(
    src: Path, dst: Path, link_target: Union[Path, str]
):
    """Atomically replace an identical materialized copy with a symbolic link."""
    if not dst.is_file() or not filecmp.cmp(src, dst, shallow=False):
        raise FileExistsError(
            f"Refusing to replace non-identical destination while linking: {dst}"
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{dst.name}.", suffix=".symlink.tmp", dir=dst.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        os.symlink(link_target, temporary)
        os.replace(temporary, dst)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def copy_or_link(src: Path, dst: Path, symlink: bool, relative_symlink: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    ensure_writable(dst.parent)
    if dst.is_symlink():
        if not dst.exists():
            raise FileNotFoundError(f"Existing symbolic link is broken: {dst}")
        if not filecmp.cmp(src.resolve(), dst.resolve(), shallow=False):
            raise FileExistsError(
                f"Existing symbolic link points to non-identical content: {dst}"
            )
        return
    if symlink:
        link_target = os.path.relpath(src.resolve(), dst.parent.resolve()) if relative_symlink else src.resolve()
        if dst.exists():
            _replace_identical_file_with_symlink(src.resolve(), dst, link_target)
        else:
            os.symlink(link_target, dst)
    elif dst.exists():
        return
    else:
        shutil.copy2(src, dst)
        ensure_writable(dst)


def replicate_split_as_links(src_root: Path, dst_root: Path, split: str):
    """Mirror one dataset split with symbolic links."""
    for obj_path in sorted(path for path in src_root.iterdir() if path.is_dir()):
        src_split = obj_path / split
        if not src_split.is_dir():
            continue
        for src_file in src_split.rglob('*'):
            if not src_file.is_file():
                continue
            dst_file = dst_root / obj_path.name / split / src_file.relative_to(src_split)
            copy_or_link(src_file, dst_file, symlink=True, relative_symlink=True)


def replicate_split(src_root: Path, dst_root: Path, split: str):
    """Copy entire `split` folder (test or ground_truth) object‑wise if missing."""
    for obj in sorted(p.name for p in src_root.iterdir() if p.is_dir()):
        src_split = src_root / obj / split
        if not src_split.is_dir():
            continue
        dst_split = dst_root / obj / split
        ensure_tree_writable(dst_split)
        shutil.copytree(src_split, dst_split, dirs_exist_ok=True)
        ensure_tree_writable(dst_split)


def gather_defect_dirs(src_root: Path) -> Dict[str, List[str]]:
    """Return {object → list of defect category names}, longest first"""
    mapping: Dict[str, List[str]] = {}
    for obj in sorted(p.name for p in src_root.iterdir() if p.is_dir()):
        test_dir = src_root / obj / "test"
        if not test_dir.is_dir():
            continue
        defects = sorted((d.name for d in test_dir.iterdir() if d.is_dir()), key=len, reverse=True)
        mapping[obj] = defects
    return mapping

###############################################################################
# Core steps
###############################################################################

def prune_good_samples(dest: Path, prune_manifest: Path):
    if not prune_manifest.is_file():
        raise FileNotFoundError(f"Prune manifest not found: {prune_manifest}")
    removed, missing = 0, 0
    with open(prune_manifest) as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            target = dest / rel
            if target.is_file():
                ensure_writable(target.parent)
                if not target.is_symlink():
                    ensure_writable(target)
                target.unlink()
                removed += 1
            else:
                missing += 1
    print(f"✔ Removed {removed} original good images (listed in prune manifest)")
    if missing:
        print(f"⚠ {missing} paths from prune manifest were not found (already absent)")


def inject_defect_samples(
    source: Path,
    dest: Path,
    noisy_manifest: Path,
    symlink: bool,
    relative_symlink: bool = False,
):
    if not noisy_manifest.is_file():
        raise FileNotFoundError(f"Noisy manifest not found: {noisy_manifest}")

    defect_map = gather_defect_dirs(source)
    injected, missing_sources = 0, []

    with open(noisy_manifest) as f:
        for rel_dst in (ln.strip() for ln in f if ln.strip()):
            parts = Path(rel_dst).parts
            if len(parts) < 4 or parts[1:3] != ("train", "good"):
                raise ValueError(f"Malformed noisy manifest entry: {rel_dst}")
            obj = parts[0]
            filename = parts[3]
            defects = defect_map.get(obj, [])
            defect = next((d for d in defects if filename.startswith(f"{d}_")), None)
            if defect is None:
                raise ValueError(f"Cannot determine defect for entry: {rel_dst}")
            original_name = filename[len(defect) + 1 :]
            src_img = source / obj / "test" / defect / original_name
            if not src_img.is_file():
                missing_sources.append(str(src_img))
                continue
            dst_img = dest / rel_dst
            copy_or_link(src_img, dst_img, symlink, relative_symlink)
            injected += 1

    if missing_sources:
        preview = "\n".join(missing_sources[:10])
        more = "..." if len(missing_sources) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing_sources)} source images missing while injecting:\n{preview}{more}"
        )
    print(f"✔ Injected {injected} defect images (listed in noisy manifest)")

###############################################################################
# CLI
###############################################################################

def main():
    p = argparse.ArgumentParser(description="Build a long‑tail noisy MVTec‑AD dataset from two manifests.")
    p.add_argument("--source-dir", required=True, type=Path, help="Pristine MVTec‑AD root directory")
    p.add_argument("--dest-dir", required=True, type=Path, help="Output root for the long‑tail noisy dataset")
    p.add_argument("--noisy-manifest", required=True, type=Path, help="Manifest of defect images to inject")
    p.add_argument("--prune-manifest", required=True, type=Path, help="Manifest of original good images to delete")
    p.add_argument("--symlink", action="store_true", help="Use symbolic links instead of copying when injecting defects")
    p.add_argument(
        "--symlink-all",
        action="store_true",
        help=(
            "Build every split with symbolic links; identical existing copies are "
            "atomically converted in place"
        ),
    )
    args = p.parse_args()

    # 1. Ensure pristine structure in dest
    if args.symlink_all:
        replicate_split_as_links(args.source_dir, args.dest_dir, "train")
        replicate_split_as_links(args.source_dir, args.dest_dir, "test")
        replicate_split_as_links(args.source_dir, args.dest_dir, "ground_truth")
    else:
        for obj_path in sorted(
            [d for d in args.source_dir.iterdir() if (d / "train").is_dir()]):
            obj_name = obj_path.name
            src_good = obj_path / "train" / "good"
            dst_good = args.dest_dir / obj_name / "train" / "good"
            ensure_tree_writable(dst_good)
            shutil.copytree(src_good, dst_good, dirs_exist_ok=True)
            ensure_tree_writable(dst_good)
        replicate_split(args.source_dir, args.dest_dir, "test")
        replicate_split(args.source_dir, args.dest_dir, "ground_truth")

    # 2. Prune specified good samples
    prune_good_samples(args.dest_dir, args.prune_manifest)

    # 3. Inject defect samples
    inject_defect_samples(
        args.source_dir,
        args.dest_dir,
        args.noisy_manifest,
        args.symlink or args.symlink_all,
        relative_symlink=args.symlink_all,
    )

    print(f"✔ Long‑tail noisy dataset ready at {args.dest_dir}")


if __name__ == "__main__":
    main()
