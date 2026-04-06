#!/usr/bin/env python3
"""Resize images, write class folders, optional validation split and S3 sync."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from datasets import load_from_disk

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from plant_disease_mlops.data.preprocess import preprocess_dataset, sync_to_s3  # noqa: E402
from plant_disease_mlops.data.transforms import inference_image_size  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=_ROOT / "data" / "raw" / "plantvillage",
        help="HuggingFace load_from_disk path",
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=_ROOT / "data" / "processed",
        help="Output root with train/validation/test",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of train held out for validation (0 to disable)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-s3",
        action="store_true",
        help="Skip S3 even if PLANT_DISEASE_S3_BUCKET is set",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip preprocessing; only run aws s3 sync from --processed to S3",
    )
    args = parser.parse_args()

    if args.upload_only and args.no_s3:
        parser.error("--upload-only cannot be used with --no-s3")

    if not args.upload_only:
        print(f"Loading dataset from {args.raw} ({inference_image_size()}px JPEGs)...")
        ds = load_from_disk(str(args.raw))
        preprocess_dataset(
            ds,
            args.processed,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        print("Preprocessing complete:", args.processed)
    else:
        if not args.processed.is_dir():
            parser.error(f"--processed must exist for --upload-only: {args.processed}")

    bucket = os.environ.get("PLANT_DISEASE_S3_BUCKET")
    prefix = os.environ.get("PLANT_DISEASE_S3_PREFIX", "processed")
    if bucket and not args.no_s3:
        sync_to_s3(args.processed, bucket, prefix)
    elif args.upload_only:
        parser.error("PLANT_DISEASE_S3_BUCKET must be set when using --upload-only.")
    elif bucket is None:
        print("PLANT_DISEASE_S3_BUCKET not set; skipping S3 sync.")


if __name__ == "__main__":
    main()
