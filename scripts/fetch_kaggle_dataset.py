"""
Fetch a Kaggle dataset for the GPU-activation / training sprint.

Real, verified integration: this script uses the official ``kaggle`` package
and your local ``~/.kaggle/kaggle.json`` credentials (already configured on
this machine — auth was confirmed working, see PROJECT_KNOWLEDGE_GRAPH /
session notes). It's deliberately a standalone script, not run automatically
by any agent/test, because dataset downloads are large (100MB-several GB) and
should be a deliberate, reviewed step before retraining.

Usage:
    python scripts/fetch_kaggle_dataset.py --search "plant disease"          # list candidates, no download
    python scripts/fetch_kaggle_dataset.py --dataset vipoooool/new-plant-diseases-dataset --dest data/kaggle/new_plant_diseases
    python scripts/fetch_kaggle_dataset.py --dataset <owner>/<slug> --dest <dir> --dry-run   # verify auth + existence only

Recommended datasets for the accuracy/training sprint (verified to exist via a
live Kaggle search on 2026-07-31):
    vipoooool/new-plant-diseases-dataset        (~87k images, PlantVillage-augmented —
                                                  likely the same source as the existing
                                                  "Wheat2" 87,867-image split per project memory)
    rashikrahmanpritom/plant-disease-recognition-dataset
    sadmansakibmahi/plant-disease-expert
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def search(query: str, limit: int = 10) -> None:
    import kaggle

    kaggle.api.authenticate()
    results = kaggle.api.dataset_list(search=query)
    print(f"Found {len(results)} datasets for '{query}' (showing up to {limit}):")
    for d in results[:limit]:
        size = getattr(d, "size", "unknown")
        print(f"  - {d.ref}  (size={size})")


def fetch(dataset_ref: str, dest: str, dry_run: bool = False) -> None:
    import kaggle

    kaggle.api.authenticate()
    if dry_run:
        # Verify the dataset exists and credentials work, without downloading.
        meta = kaggle.api.dataset_list(search=dataset_ref)
        matches = [d for d in meta if d.ref == dataset_ref]
        if not matches:
            print(f"NOT FOUND: {dataset_ref}", file=sys.stderr)
            sys.exit(1)
        print(f"OK (dry-run): '{dataset_ref}' exists and credentials are valid. "
              f"Re-run without --dry-run to download to {dest}.")
        return

    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset_ref} -> {out} (this may take a while for large datasets)...")
    kaggle.api.dataset_download_files(dataset_ref, path=str(out), unzip=True, quiet=False)
    print(f"Done: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search", help="Search Kaggle datasets by keyword (no download)")
    parser.add_argument("--dataset", help="Dataset ref to fetch, e.g. owner/slug")
    parser.add_argument("--dest", default="data/kaggle/dataset", help="Destination directory")
    parser.add_argument("--dry-run", action="store_true", help="Verify auth + existence only, skip download")
    args = parser.parse_args()

    if args.search:
        search(args.search)
    elif args.dataset:
        fetch(args.dataset, args.dest, dry_run=args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
