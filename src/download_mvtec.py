"""MVTec AD Dataset Downloader and Integrity Verifier.

Features:
- Checks if local MVTec AD files already exist.
- Displays explicit licensing terms and citation details (CC BY-NC-SA 4.0).
- Provides exact manual download instructions for official MVTec website archives.
- Automated fast concurrent downloader for individual categories (e.g., bottle, metal_nut).
- Preserves standard folder hierarchy:
    category/
        train/good/
        test/good/, test/<defect_types>/
        ground_truth/<defect_types>/
"""

import os
import sys
import json
import shutil
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


HF_TREE_API = "https://huggingface.co/api/datasets/foersben/mvtec-ad/tree/main/{subpath}"
HF_RAW_BASE = "https://huggingface.co/datasets/foersben/mvtec-ad/resolve/main/{filepath}"

LICENSE_TEXT = """MVTec Anomaly Detection Dataset (MVTec AD)
License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
Copyright: 2019 MVTec Software GmbH (https://www.mvtec.com)

If you use this dataset in scientific or commercial projects, please cite:
Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger,
"A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection",
IEEE Conference on Computer Vision and Pattern Recognition, 2019.
"""


def check_dataset_exists(category_dir: Path) -> dict:
    """Checks whether the category directory exists and has the expected structure."""
    if not category_dir.is_dir():
        return {"exists": False, "reason": f"Directory {category_dir} does not exist."}

    train_good = category_dir / "train" / "good"
    test_dir = category_dir / "test"
    gt_dir = category_dir / "ground_truth"

    if not train_good.is_dir():
        return {"exists": False, "reason": "Missing train/good directory."}
    if not test_dir.is_dir():
        return {"exists": False, "reason": "Missing test directory."}

    train_imgs = list(train_good.glob("*.png")) + list(train_good.glob("*.jpg"))
    test_imgs = list(test_dir.glob("*/*.png")) + list(test_dir.glob("*/*.jpg"))
    gt_imgs = list(gt_dir.glob("*/*.png")) + list(gt_dir.glob("*/*.jpg")) if gt_dir.is_dir() else []

    if len(train_imgs) == 0:
        return {"exists": False, "reason": "train/good directory is empty."}
    if len(test_imgs) == 0:
        return {"exists": False, "reason": "test directory contains no images."}

    return {
        "exists": True,
        "train_count": len(train_imgs),
        "test_count": len(test_imgs),
        "gt_count": len(gt_imgs),
        "test_subdirs": [p.name for p in test_dir.iterdir() if p.is_dir()]
    }


def print_manual_instructions(category: str, target_dir: Path):
    """Outputs exact manual steps for downloading directly from official MVTec website."""
    print("=" * 78)
    print("MVTec AD DATASET MANUAL DOWNLOAD & SETUP INSTRUCTIONS")
    print("=" * 78)
    print(LICENSE_TEXT)
    print("-" * 78)
    print("Step 1: Visit the official MVTec research dataset portal:")
    print("        https://www.mvtec.com/company/research/datasets/mvtec-ad")
    print("Step 2: Read and accept the dataset license terms on the MVTec website.")
    print("Step 3: Download the dataset archive (or the category archive if available).")
    print(f"Step 4: Extract the folder '{category}' to the destination path:")
    print(f"        {target_dir.resolve()}")
    print("Step 5: Verify the extracted structure matches:")
    print(f"        {target_dir.resolve()}/")
    print("        ├── train/")
    print("        │   └── good/")
    print("        ├── test/")
    print("        │   ├── good/")
    print("        │   └── <defect_types>/")
    print("        └── ground_truth/")
    print("            └── <defect_types>/")
    print("=" * 78)


def _fetch_tree_files(subpath: str) -> list:
    """Recursively fetches all file paths from the Hugging Face repository tree."""
    url = HF_TREE_API.format(subpath=subpath)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            items = json.loads(res.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to query dataset metadata from {url}: {e}")

    file_paths = []
    for item in items:
        item_path = item["path"]
        item_type = item.get("type", "file")
        if item_type == "directory":
            file_paths.extend(_fetch_tree_files(item_path))
        else:
            file_paths.append(item_path)
    return file_paths


def _download_single_file(remote_path: str, local_path: Path):
    """Downloads a single file from raw repository URL."""
    url = HF_RAW_BASE.format(filepath=remote_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Skip if file already exists with nonzero size
    if local_path.is_file() and local_path.stat().st_size > 0:
        return True

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            content = res.read()
            with open(local_path, "wb") as f:
                f.write(content)
        return True
    except Exception as e:
        if local_path.is_file():
            local_path.unlink()
        raise RuntimeError(f"Error downloading {remote_path}: {e}")


def download_category(category: str, base_dir: Path, max_workers: int = 12) -> bool:
    """Automated concurrent downloader for a specific MVTec category."""
    category_dir = base_dir / category
    status = check_dataset_exists(category_dir)
    if status["exists"]:
        print(f"[Dataset] Verified existing files for '{category}' at {category_dir}")
        print(f"          - Train normal images: {status['train_count']}")
        print(f"          - Test images:         {status['test_count']} (Categories: {status['test_subdirs']})")
        print(f"          - Ground truth masks:  {status['gt_count']}")
        return True

    print(f"\n[Dataset] Dataset not found at {category_dir}.")
    print(LICENSE_TEXT)
    print(f"[Dataset] Fetching file catalog for '{category}'...")

    try:
        all_files = _fetch_tree_files(category)
    except Exception as err:
        print(f"[Dataset Error] Failed to fetch remote file catalog: {err}")
        print_manual_instructions(category, category_dir)
        return False

    print(f"[Dataset] Discovered {len(all_files)} files to download for '{category}'.")
    print(f"[Dataset] Commencing concurrent download ({max_workers} worker threads)...")

    category_dir.mkdir(parents=True, exist_ok=True)
    
    # Save license and citation files locally
    (category_dir / "LICENSE.txt").write_text(LICENSE_TEXT, encoding="utf-8")

    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for remote_rel in all_files:
            # relative path inside category folder: remote_rel is e.g. "bottle/train/good/000.png"
            sub_rel = Path(remote_rel).relative_to(category)
            local_dst = category_dir / sub_rel
            futures[executor.submit(_download_single_file, remote_rel, local_dst)] = remote_rel

        with tqdm(total=len(all_files), desc=f"Downloading {category}", unit="file") as pbar:
            for future in as_completed(futures):
                rel = futures[future]
                try:
                    future.result()
                    success_count += 1
                except Exception as ex:
                    print(f"\n[Warning] Failed to download {rel}: {ex}")
                pbar.update(1)

    # Post-download verification
    post_check = check_dataset_exists(category_dir)
    if post_check["exists"]:
        print(f"\n[Dataset] Successfully downloaded and verified '{category}'!")
        print(f"          - Train images: {post_check['train_count']}")
        print(f"          - Test images:  {post_check['test_count']}")
        print(f"          - Masks:        {post_check['gt_count']}")
        return True
    else:
        print(f"\n[Dataset] Download completed with issues: {post_check['reason']}")
        print_manual_instructions(category, category_dir)
        return False


def main():
    parser = argparse.ArgumentParser(description="MVTec AD Dataset Download & Verification")
    parser.add_argument("--category", type=str, default="bottle", help="Category name (e.g. bottle, metal_nut)")
    parser.add_argument("--data_dir", type=str, default="data/mvtec", help="Destination base directory")
    parser.add_argument("--check_only", action="store_true", help="Only check existing local files")
    parser.add_argument("--workers", type=int, default=12, help="Concurrent download threads")
    args = parser.parse_args()

    base_path = Path(args.data_dir)
    cat_path = base_path / args.category

    if args.check_only:
        status = check_dataset_exists(cat_path)
        if status["exists"]:
            print(f"[OK] Category '{args.category}' is present and valid:")
            print(f"     Train images: {status['train_count']}")
            print(f"     Test images:  {status['test_count']}")
            print(f"     GT masks:     {status['gt_count']}")
            sys.exit(0)
        else:
            print(f"[NOT FOUND] Category '{args.category}' not ready: {status['reason']}")
            print_manual_instructions(args.category, cat_path)
            sys.exit(1)

    ok = download_category(args.category, base_path, max_workers=args.workers)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
