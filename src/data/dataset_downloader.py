"""
Production Mendeley Dataset Downloader
=======================================
Directly connects to Mendeley Data REST API (DOI: 10.17632/8ckhkssyn3)
and downloads genuine Indian currency notes across all 7 denominations:
10, 20, 50, 100, 200, 500, 2000.
"""

import os
import sys
import time
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image

DATASET_ID = "8ckhkssyn3"
VERSION = "1"
BASE_URL = "https://data.mendeley.com/public-api"
HEADERS = {
    "Accept": "application/vnd.mendeley-public-dataset.1+json"
}
DENOMINATIONS = [10, 20, 50, 100, 200, 500, 2000]


def get_folders():
    """Retrieve denomination folders from Mendeley."""
    url = f"{BASE_URL}/datasets/{DATASET_ID}/folders/{VERSION}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    folders = resp.json()
    
    folder_map = {}
    for f in folders:
        name = f.get("name", "").strip()
        try:
            denom = int(name)
            if denom in DENOMINATIONS:
                folder_map[denom] = f.get("id")
        except ValueError:
            pass
    return folder_map


def get_files_in_folder(folder_id):
    """Retrieve all files contained in a denomination folder."""
    url = f"{BASE_URL}/datasets/{DATASET_ID}/files?folder_id={folder_id}&version={VERSION}&$start=0&$limit=1000"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_single_image(item, denom, target_dir, max_size=(800, 800)):
    """Downloads one banknote image, converts to RGB, resizes, and saves to disk."""
    fname = item.get("filename", "")
    content_details = item.get("content_details", {})
    download_url = content_details.get("download_url")
    
    if not download_url:
        return False, denom, fname
        
    out_file = Path(target_dir) / f"{denom}_{fname}"
    if out_file.exists() and out_file.stat().st_size > 1000:
        return True, denom, fname

    retries = 3
    for _ in range(retries):
        try:
            r = requests.get(download_url, timeout=30)
            if r.status_code == 200:
                img = Image.open(io.BytesIO(r.content))
                img = img.convert("RGB")
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                img.save(out_file, "JPEG", quality=90)
                return True, denom, fname
        except Exception:
            time.sleep(1)
            
    return False, denom, fname


def download_mendeley_dataset(
    output_dir: str = "data/raw",
    samples_per_denom: int = 30,
    max_workers: int = 8
):
    """Downloads a balanced set of genuine Indian currency notes."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Querying Mendeley dataset folder tree...")
    folder_map = get_folders()
    print(f"Found {len(folder_map)} denomination folders: {sorted(folder_map.keys())}")

    tasks = []
    for denom in sorted(folder_map.keys()):
        folder_id = folder_map[denom]
        files = get_files_in_folder(folder_id)
        selected_files = files[:samples_per_denom]
        print(f"Denomination {denom}: Total available {len(files)}, Selected {len(selected_files)}")
        
        denom_dir = output_path / str(denom)
        denom_dir.mkdir(parents=True, exist_ok=True)
        
        for item in selected_files:
            tasks.append((item, denom, denom_dir))

    print(f"\nStarting parallel download for {len(tasks)} images with {max_workers} threads...")
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single_image, item, denom, d_dir): (denom, item) for item, denom, d_dir in tasks}
        for fut in as_completed(futures):
            ok, denom, fname = fut.result()
            if ok:
                completed += 1
                if completed % 20 == 0 or completed == len(tasks):
                    print(f"Downloaded {completed}/{len(tasks)} images...")

    print(f"\nGenuine dataset download complete! {completed} images saved across all 7 denominations in '{output_dir}'.")
    return completed


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    download_mendeley_dataset("data/raw", samples_per_denom=count, max_workers=8)
