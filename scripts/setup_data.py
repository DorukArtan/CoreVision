"""
setup_data.py — Extract and Organize Datasets from ZIP Files

Usage:
    python scripts/setup_data.py --cars path/to/stanford_cars.zip --plates path/to/turkish_plates.zip

This script will:
1. Extract the ZIP files
2. Auto-detect the internal folder structure
3. Reorganize into the expected format for training
"""

import os
import sys
import zipfile
import shutil
import argparse
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def extract_zip(zip_path, extract_to):
    """Extract a ZIP file to the target directory."""
    zip_path = Path(zip_path).resolve()
    if not zip_path.exists():
        print(f"  ERROR: ZIP file not found: {zip_path}")
        return False

    print(f"  Extracting: {zip_path.name}")
    print(f"  Target:     {extract_to}")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # List top-level contents
        top_level = set()
        for name in zf.namelist():
            parts = name.split('/')
            if parts[0]:
                top_level.add(parts[0])

        print(f"  Top-level items in ZIP: {sorted(top_level)}")
        zf.extractall(extract_to)

    print(f"  ✅ Extracted successfully!")
    return True


def find_image_dirs(root):
    """Recursively find directories containing images."""
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    result = []
    for dirpath, dirnames, filenames in os.walk(root):
        imgs = [f for f in filenames if Path(f).suffix.lower() in image_exts]
        if imgs:
            result.append((dirpath, len(imgs)))
    return result


def find_label_dirs(root):
    """Find directories containing .txt label files (YOLO format)."""
    result = []
    for dirpath, dirnames, filenames in os.walk(root):
        txts = [f for f in filenames if f.endswith('.txt')]
        if txts:
            result.append((dirpath, len(txts)))
    return result


def setup_stanford_cars(zip_path):
    """
    Extract and organize Stanford Cars dataset.

    Expected output structure:
        data/stanford_cars/
        ├── train/
        │   ├── ClassName1/
        │   │   ├── image1.jpg
        │   │   └── ...
        │   └── ClassName2/
        │       └── ...
        └── test/
            └── (same structure)
    """
    print("\n" + "=" * 60)
    print("STANFORD CARS DATASET")
    print("=" * 60)

    target_dir = DATA_DIR / "stanford_cars"
    temp_dir = DATA_DIR / "_temp_cars"

    # Extract to temp
    os.makedirs(temp_dir, exist_ok=True)
    if not extract_zip(zip_path, temp_dir):
        return False

    # Analyze structure
    print("\n  Analyzing extracted structure...")
    img_dirs = find_image_dirs(temp_dir)

    if not img_dirs:
        print("  ERROR: No images found in the ZIP!")
        return False

    print(f"  Found {len(img_dirs)} directories with images:")
    for d, count in sorted(img_dirs, key=lambda x: -x[1])[:10]:
        rel = os.path.relpath(d, temp_dir)
        print(f"    {rel}: {count} images")
    if len(img_dirs) > 10:
        print(f"    ... and {len(img_dirs) - 10} more")

    # Try to detect the structure
    # Case 1: Already organized as train/test with class subdirectories
    # Case 2: Flat with train/ and test/ folders
    # Case 3: Single folder with all images

    os.makedirs(target_dir, exist_ok=True)

    # Check if there's a nested root folder (common in ZIPs)
    temp_contents = os.listdir(temp_dir)
    actual_root = temp_dir
    if len(temp_contents) == 1 and os.path.isdir(temp_dir / temp_contents[0]):
        actual_root = temp_dir / temp_contents[0]
        print(f"  Detected nested root: {temp_contents[0]}/")

    # Check if train/test split exists
    has_train = any(
        'train' in os.path.relpath(d, actual_root).lower().split(os.sep)
        for d, _ in img_dirs
    )
    has_test = any(
        'test' in os.path.relpath(d, actual_root).lower().split(os.sep)
        for d, _ in img_dirs
    )

    if has_train and has_test:
        print("  ✅ Detected train/test split in the dataset!")
        # Move the entire structure
        if actual_root != target_dir:
            # Copy contents to target
            for item in os.listdir(actual_root):
                src = actual_root / item
                dst = target_dir / item
                if src.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
    else:
        print("  ⚠️  No train/test split detected. Moving all contents to target.")
        for item in os.listdir(actual_root):
            src = actual_root / item
            dst = target_dir / item
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    # Cleanup temp
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Final verification
    print(f"\n  Final structure at: {target_dir}")
    _print_tree(target_dir, max_depth=2)
    
    total_images = sum(c for _, c in find_image_dirs(target_dir))
    print(f"\n  Total images: {total_images}")

    return True


def setup_turkish_plates(zip_path):
    """
    Extract and organize Turkish Plates dataset.

    Expected output structure (YOLO format):
        data/turkish_plates/
        ├── images/
        │   ├── train/
        │   │   ├── img001.jpg
        │   │   └── ...
        │   └── val/
        │       └── ...
        └── labels/
            ├── train/
            │   ├── img001.txt
            │   └── ...
            └── val/
                └── ...
    """
    print("\n" + "=" * 60)
    print("TURKISH LICENSE PLATE DATASET")
    print("=" * 60)

    target_dir = DATA_DIR / "turkish_plates"
    temp_dir = DATA_DIR / "_temp_plates"

    # Extract to temp
    os.makedirs(temp_dir, exist_ok=True)
    if not extract_zip(zip_path, temp_dir):
        return False

    # Analyze structure
    print("\n  Analyzing extracted structure...")
    img_dirs = find_image_dirs(temp_dir)
    label_dirs = find_label_dirs(temp_dir)

    print(f"  Image directories: {len(img_dirs)}")
    for d, count in img_dirs[:5]:
        rel = os.path.relpath(d, temp_dir)
        print(f"    {rel}: {count} images")

    print(f"  Label directories: {len(label_dirs)}")
    for d, count in label_dirs[:5]:
        rel = os.path.relpath(d, temp_dir)
        print(f"    {rel}: {count} labels")

    os.makedirs(target_dir, exist_ok=True)

    # Check if there's a nested root folder
    temp_contents = os.listdir(temp_dir)
    actual_root = temp_dir
    if len(temp_contents) == 1 and os.path.isdir(temp_dir / temp_contents[0]):
        actual_root = temp_dir / temp_contents[0]
        print(f"  Detected nested root: {temp_contents[0]}/")

    # Check for YOLO structure (images/ and labels/ folders)
    has_images_dir = any('images' in os.listdir(actual_root)) if os.path.exists(actual_root) else False
    has_labels_dir = any('labels' in os.listdir(actual_root)) if os.path.exists(actual_root) else False

    if has_images_dir and has_labels_dir:
        print("  ✅ Detected YOLO directory structure (images/ + labels/)")
    else:
        print("  ℹ️  Structure may differ from standard YOLO format")

    # Copy everything to target
    for item in os.listdir(actual_root):
        src = actual_root / item
        dst = target_dir / item
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # Cleanup temp
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Final verification
    print(f"\n  Final structure at: {target_dir}")
    _print_tree(target_dir, max_depth=3)

    total_images = sum(c for _, c in find_image_dirs(target_dir))
    total_labels = sum(c for _, c in find_label_dirs(target_dir))
    print(f"\n  Total images: {total_images}")
    print(f"  Total labels: {total_labels}")

    # Verify pairing
    if total_images > 0 and total_labels > 0:
        print(f"  Image/Label ratio: {total_labels/total_images:.2%}")
        if abs(total_images - total_labels) < total_images * 0.1:
            print("  ✅ Images and labels appear to match!")
        else:
            print("  ⚠️  Image and label counts differ significantly — check your data")

    return True


def _print_tree(path, prefix="  ", max_depth=2, _depth=0):
    """Print a directory tree."""
    if _depth >= max_depth:
        return

    path = Path(path)
    items = sorted(path.iterdir())
    dirs = [i for i in items if i.is_dir()]
    files = [i for i in items if i.is_file()]

    for d in dirs:
        child_count = sum(1 for _ in d.rglob('*') if _.is_file())
        print(f"{prefix}📁 {d.name}/ ({child_count} files)")
        _print_tree(d, prefix + "  ", max_depth, _depth + 1)

    if files and _depth < max_depth:
        if len(files) <= 3:
            for f in files:
                print(f"{prefix}📄 {f.name}")
        else:
            print(f"{prefix}📄 {files[0].name} ... +{len(files)-1} more files")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and organize datasets from ZIP files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup_data.py --cars ~/Downloads/stanford_cars.zip --plates ~/Downloads/plates.zip
  python scripts/setup_data.py --cars C:\\Users\\USER\\Downloads\\cars.zip
  python scripts/setup_data.py --plates C:\\Users\\USER\\Downloads\\plates.zip
        """
    )
    parser.add_argument('--cars', type=str, help='Path to Stanford Cars ZIP file')
    parser.add_argument('--plates', type=str, help='Path to Turkish Plates ZIP file')

    args = parser.parse_args()

    if not args.cars and not args.plates:
        parser.print_help()
        print("\n⚠️  Please provide at least one ZIP file path!")
        sys.exit(1)

    print("=" * 60)
    print("Dataset Setup Script")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.cars:
        setup_stanford_cars(args.cars)

    if args.plates:
        setup_turkish_plates(args.plates)

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Verify the data directory structure looks correct above")
    print(f"  2. Run: python test_model.py")
    print(f"  3. Start training: python -m training.train")


if __name__ == "__main__":
    main()
