#!/usr/bin/env python3
"""
Prepare Omni3D dataset directories for this repository.

This script does not download Omni3D directly. Follow the official dataset
instructions first, then use this helper to place/link data into:
  - data/omni3d/images
  - data/omni3d/annotations
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

OMNI3D_URL = "https://github.com/facebookresearch/omni3d#dataset"


def _count_image_files(path: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _count_json_files(path: Path) -> int:
    return sum(1 for p in path.rglob("*.json") if p.is_file())


def _remove_if_exists(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _materialize(src: Path, dst: Path, mode: str, force: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not force:
            raise FileExistsError(
                f"Destination already exists: {dst}. Use --force to replace it."
            )
        _remove_if_exists(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "link":
        dst.symlink_to(src.resolve(), target_is_directory=True)
    elif mode == "copy":
        shutil.copytree(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _print_summary(images_dst: Path, annotations_dst: Path) -> None:
    image_count = _count_image_files(images_dst)
    json_count = _count_json_files(annotations_dst)

    print("\nOmni3D layout is ready:")
    print(f"  images:      {images_dst} ({image_count} image files)")
    print(f"  annotations: {annotations_dst} ({json_count} json files)")

    if image_count == 0:
        print("\nWARNING: No image files detected under the images directory.")
    if json_count == 0:
        print("WARNING: No annotation json files detected under the annotations directory.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare data/omni3d layout from your downloaded Omni3D dataset"
    )
    parser.add_argument(
        "--source-images",
        required=False,
        help="Path to downloaded Omni3D images directory",
    )
    parser.add_argument(
        "--source-annotations",
        required=True,
        help="Path to downloaded Omni3D annotations directory",
    )
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Set up only annotations when image assets are not available yet",
    )
    parser.add_argument(
        "--mode",
        choices=["link", "copy"],
        default="link",
        help="How to place data in this repo (default: link)",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to this repository root (default: current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing data/omni3d/images and data/omni3d/annotations",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    annotations_src = Path(args.source_annotations).resolve()
    images_src = Path(args.source_images).resolve() if args.source_images else None

    if not args.annotations_only:
        if images_src is None:
            print("ERROR: --source-images is required unless --annotations-only is set")
            sys.exit(1)
        if not images_src.exists() or not images_src.is_dir():
            print(f"ERROR: source images directory not found: {images_src}")
            print(f"Download instructions: {OMNI3D_URL}")
            sys.exit(1)

    if not annotations_src.exists() or not annotations_src.is_dir():
        print(f"ERROR: source annotations directory not found: {annotations_src}")
        print(f"Download instructions: {OMNI3D_URL}")
        sys.exit(1)

    omni_root = repo_root / "data" / "omni3d"
    images_dst = omni_root / "images"
    annotations_dst = omni_root / "annotations"

    omni_root.mkdir(parents=True, exist_ok=True)

    try:
        if images_src is not None:
            _materialize(images_src, images_dst, args.mode, args.force)
        else:
            images_dst.mkdir(parents=True, exist_ok=True)
        _materialize(annotations_src, annotations_dst, args.mode, args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        sys.exit(1)

    _print_summary(images_dst, annotations_dst)

    print("\nNext steps:")
    print("  1) Build or update your test_cases JSON with image paths in data/omni3d/images")
    print("  2) Run evaluation:")
    print("     python evaluate_benchmark.py --test-cases data/omni3d/test_cases.json --output-dir outputs/omni3d")


if __name__ == "__main__":
    main()
