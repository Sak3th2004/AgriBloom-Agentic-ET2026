"""
Dataset Preparation — Unify ALL Indian crop disease datasets into standard format.
Covers: Cotton, Rice, Wheat, Maize, Sugarcane, Ragi, Tomato, Potato, Pepper
Total: 50+ disease classes from real Indian agricultural data.

Usage:
    python data/prepare_dataset.py
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_dataset")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
UNIFIED_DIR = ROOT / "data" / "unified"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _copy_images(src_dir: Path, dest_dir: Path, max_per_class: int = 5000) -> int:
    """Copy images from src to dest directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src_dir.iterdir():
        if f.suffix.lower() in IMAGE_EXTS:
            if count >= max_per_class:
                break
            dest_file = dest_dir / f.name
            if not dest_file.exists():
                shutil.copy2(f, dest_file)
                count += 1
    return count


# ── Class name mapping for the Indian multi-crop dataset (data/raw/wheat/Train) ─
# This maps raw folder names to our unified class naming: crop___disease
INDIAN_MULTICROP_MAP = {
    # Cotton diseases — CRITICAL for Indian farmers
    "American Bollworm on Cotton": "cotton___american_bollworm",
    "Anthracnose on Cotton": "cotton___anthracnose",
    "bacterial_blight in Cotton": "cotton___bacterial_blight",
    "bollrot on Cotton": "cotton___boll_rot",
    "bollworm on Cotton": "cotton___bollworm",
    "Cotton Aphid": "cotton___aphid",
    "cotton mealy bug": "cotton___mealy_bug",
    "cotton whitefly": "cotton___whitefly",
    "Healthy cotton": "cotton___healthy",
    "Leaf Curl": "cotton___leaf_curl",
    "pink bollworm in cotton": "cotton___pink_bollworm",
    "red cotton bug": "cotton___red_bug",
    "thirps on  cotton": "cotton___thrips",

    # Rice diseases
    "Becterial Blight in Rice": "rice___bacterial_blight",
    "Rice Blast": "rice___blast",
    "Brownspot": "rice___brown_spot",
    "Tungro": "rice___tungro",
    "Leaf smut": "rice___leaf_smut",

    # Maize diseases
    "Common_Rust": "maize___common_rust",
    "Gray_Leaf_Spot": "maize___gray_leaf_spot",
    "Healthy Maize": "maize___healthy",
    "maize ear rot": "maize___ear_rot",
    "maize fall armyworm": "maize___fall_armyworm",
    "maize stem borer": "maize___stem_borer",

    # Sugarcane diseases
    "Mosaic sugarcane": "sugarcane___mosaic",
    "RedRot sugarcane": "sugarcane___red_rot",
    "RedRust sugarcane": "sugarcane___red_rust",
    "Sugarcane Healthy": "sugarcane___healthy",
    "Yellow Rust Sugarcane": "sugarcane___yellow_rust",
    "Wilt": "sugarcane___wilt",

    # Wheat diseases
    "Flag Smut": "wheat___flag_smut",
    "Healthy Wheat": "wheat___healthy",
    "Wheat aphid": "wheat___aphid",
    "Wheat black rust": "wheat___black_rust",
    "Wheat Brown leaf Rust": "wheat___brown_leaf_rust",
    "Wheat leaf blight": "wheat___leaf_blight",
    "Wheat mite": "wheat___mite",
    "Wheat powdery mildew": "wheat___powdery_mildew",
    "Wheat scab": "wheat___scab",
    "Wheat Stem fly": "wheat___stem_fly",
    "Wheat___Yellow_Rust": "wheat___yellow_rust",

    # General pest (can affect multiple crops)
    "Army worm": "general___army_worm",
}


def prepare_indian_multicrop():
    """Process the Indian multi-crop dataset (stored in data/raw/wheat/)."""
    for split in ["Train", "Validation"]:
        src_dir = RAW_DIR / "wheat" / split
        if not src_dir.exists():
            continue

        logger.info(f"Processing Indian multi-crop dataset ({split})...")
        total = 0

        for class_dir in sorted(src_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            # Map to unified class name
            unified_name = INDIAN_MULTICROP_MAP.get(class_dir.name)
            if not unified_name:
                # Try fuzzy match
                for key, val in INDIAN_MULTICROP_MAP.items():
                    if key.lower() == class_dir.name.lower():
                        unified_name = val
                        break

            if not unified_name:
                logger.warning(f"  Skipping unmapped class: {class_dir.name}")
                continue

            dest = UNIFIED_DIR / unified_name
            count = _copy_images(class_dir, dest)
            if count > 0:
                total += count
                logger.info(f"  {class_dir.name} → {unified_name}: {count} images")

        logger.info(f"Indian multi-crop ({split}) total: {total} images")


def prepare_plantvillage():
    """Process PlantVillage dataset (Tomato, Potato, Pepper)."""
    pv_dir = RAW_DIR / "plantvillage"
    if not pv_dir.exists():
        # Also check wheat2 which may contain PlantVillage
        pv_dir = RAW_DIR / "wheat2"
        if not pv_dir.exists():
            logger.warning("PlantVillage not found")
            return

    logger.info("Processing PlantVillage dataset...")
    total = 0

    # Search in multiple possible structures
    search_dirs = []
    for root_dir in [pv_dir]:
        search_dirs.append(root_dir)
        for sub in root_dir.rglob("*"):
            if sub.is_dir() and sub.name.startswith(("Tomato", "Potato", "Pepper", "Corn")):
                search_dirs.append(sub.parent)
                break

    seen = set()
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for class_dir in sorted(search_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            name = class_dir.name
            # Only include tomato, potato, pepper
            name_lower = name.lower()
            if not any(name_lower.startswith(c) for c in ["tomato", "potato", "pepper", "corn"]):
                continue

            # Standardize: "Tomato_Early_blight" → "tomato___early_blight"
            # Handle various separators
            unified = name.lower().replace("__", "___")
            if "___" not in unified:
                unified = unified.replace("_", "___", 1)

            # Clean up
            unified = unified.replace(" ", "_")

            if unified in seen:
                continue
            seen.add(unified)

            dest = UNIFIED_DIR / unified
            count = _copy_images(class_dir, dest)
            if count > 0:
                total += count
                logger.info(f"  {name} → {unified}: {count} images")

    logger.info(f"PlantVillage total: {total} images")


def prepare_rice_detailed():
    """Process detailed Rice dataset with more classes."""
    rice_dir = RAW_DIR / "rice"
    if not rice_dir.exists():
        return

    logger.info("Processing Rice detailed dataset...")
    total = 0

    # Find class directories (may be nested)
    for search_dir in [rice_dir]:
        for sub in search_dir.rglob("*"):
            if sub.is_dir() and any(sub.glob("*.jpg")) or any(sub.glob("*.png")):
                # This is a leaf directory with images
                class_name = sub.name.lower().replace(" ", "_")
                unified = f"rice___{class_name}"

                dest = UNIFIED_DIR / unified
                count = _copy_images(sub, dest)
                if count > 0:
                    total += count
                    logger.info(f"  {sub.name} → {unified}: {count} images")

    logger.info(f"Rice detailed total: {total} images")


def prepare_ragi():
    """Process Ragi/Finger Millet dataset."""
    ragi_dir = RAW_DIR / "ragi"
    if not ragi_dir.exists():
        return

    logger.info("Processing Ragi dataset...")
    total = 0

    for sub in ragi_dir.rglob("*"):
        if sub.is_dir() and (any(sub.glob("*.jpg")) or any(sub.glob("*.png")) or any(sub.glob("*.JPG"))):
            class_name = sub.name.lower().replace(" ", "_")
            unified = f"ragi___{class_name}"

            dest = UNIFIED_DIR / unified
            count = _copy_images(sub, dest)
            if count > 0:
                total += count
                logger.info(f"  {sub.name} → {unified}: {count} images")

    logger.info(f"Ragi total: {total} images")


def prepare_sugarcane_standalone():
    """Process standalone Sugarcane dataset."""
    sugar_dir = RAW_DIR / "sugarcane"
    if not sugar_dir.exists():
        return

    logger.info("Processing Sugarcane standalone dataset...")
    total = 0

    for class_dir in sorted(sugar_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name.lower().replace(" ", "_")
        unified = f"sugarcane___{class_name}"

        dest = UNIFIED_DIR / unified
        count = _copy_images(class_dir, dest)
        if count > 0:
            total += count
            logger.info(f"  {class_dir.name} → {unified}: {count} images")

    logger.info(f"Sugarcane standalone total: {total} images")


def prepare_wheat_leaf():
    """Process specific wheat leaf disease dataset."""
    wl_dir = RAW_DIR / "wheat" / "wheat_leaf"
    if not wl_dir.exists():
        return

    logger.info("Processing Wheat leaf dataset...")
    total = 0

    for class_dir in sorted(wl_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name.lower().replace(" ", "_")
        unified = f"wheat___{class_name}"

        dest = UNIFIED_DIR / unified
        count = _copy_images(class_dir, dest)
        if count > 0:
            total += count
            logger.info(f"  {class_dir.name} → {unified}: {count} images")

    logger.info(f"Wheat leaf total: {total} images")


def verify_dataset():
    """Verify unified dataset and print statistics."""
    if not UNIFIED_DIR.exists():
        logger.error("Unified dataset directory does not exist!")
        return 0, 0

    classes = sorted([d for d in UNIFIED_DIR.iterdir() if d.is_dir()])
    total_images = 0
    class_stats = []
    crop_counts = {}

    for class_dir in classes:
        count = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        total_images += count
        class_stats.append((class_dir.name, count))

        crop = class_dir.name.split("___")[0]
        crop_counts[crop] = crop_counts.get(crop, 0) + count

    logger.info("=" * 70)
    logger.info("🌾 UNIFIED DATASET STATISTICS")
    logger.info("=" * 70)
    logger.info(f"Total classes: {len(classes)}")
    logger.info(f"Total images:  {total_images}")

    if class_stats:
        counts = [c for _, c in class_stats]
        logger.info(f"Min images/class: {min(counts)}")
        logger.info(f"Max images/class: {max(counts)}")
        logger.info(f"Mean images/class: {sum(counts)/len(counts):.0f}")

    logger.info("\n📊 Images per crop:")
    for crop, count in sorted(crop_counts.items()):
        logger.info(f"  {crop:15s}: {count:6d} images")

    logger.info("\n📋 All classes:")
    for name, count in class_stats:
        logger.info(f"  {name:50s}: {count:5d}")

    return len(classes), total_images


def main():
    """Run full dataset preparation pipeline."""
    logger.info("=" * 70)
    logger.info("🌾 AgriBloom — Dataset Preparation for Indian Crops")
    logger.info("=" * 70)

    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Indian multi-crop dataset (cotton, rice, wheat, maize, sugarcane)
    prepare_indian_multicrop()

    # 2. PlantVillage (tomato, potato, pepper)
    prepare_plantvillage()

    # 3. Detailed rice dataset
    prepare_rice_detailed()

    # 4. Ragi/Finger millet
    prepare_ragi()

    # 5. Standalone sugarcane
    prepare_sugarcane_standalone()

    # 6. Wheat leaf specific
    prepare_wheat_leaf()

    # Verify final dataset
    num_classes, total = verify_dataset()

    if total == 0:
        logger.error("\n❌ No images found!")
        return

    logger.info(f"\n✅ Dataset ready: {num_classes} classes, {total} images")
    logger.info(f"   Path: {UNIFIED_DIR}")
    logger.info(f"\n🚀 To train the model, run:")
    logger.info(f"   python models/train_model.py --data_dir data/unified --epochs 50 --batch_size 32")


if __name__ == "__main__":
    main()
