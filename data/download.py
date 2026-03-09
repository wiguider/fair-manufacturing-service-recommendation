"""Download and prepare datasets for the fair manufacturing recommendation project."""

import os
import sys
import zipfile
import requests
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"

DATASETS = {
    "dataco": {
        "urls": [
            "https://data.mendeley.com/public-files/datasets/8gx2fvg2k6/files/dataset.zip",
            "https://archive.ics.uci.edu/static/public/598/dataco+smart+supply+chain+for+big+data+analysis.zip",
        ],
        "dest": RAW_DIR / "dataco",
        "description": "DataCo Smart Supply Chain Dataset",
        "manual_instructions": (
            "If auto-download fails, download manually from:\n"
            "  Kaggle: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis\n"
            "  Mendeley: https://data.mendeley.com/datasets/8gx2fvg2k6/5\n"
            "Place CSV files in data/raw/dataco/"
        ),
    },
    "ai4i": {
        "urls": [
            "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
        ],
        "dest": RAW_DIR / "ai4i",
        "description": "AI4I 2020 Predictive Maintenance Dataset",
        "manual_instructions": None,
    },
}

MSKG_INFO = """
=== MSKG Dataset ===
The Manufacturing Service Knowledge Graph (MSKG) must be obtained from:
  Paper: "Building A Knowledge Graph to Enrich ChatGPT Responses in Manufacturing Service Discovery"
  arXiv: 2404.06571 (Li & Starly, 2024)

Please download the MSKG data and place it in: data/raw/mskg/

Expected files:
  - manufacturers.csv (or .json)
  - services.csv
  - certifications.csv
  - locations.csv
  - relationships.csv (manufacturer-service edges)
"""


def download_file(urls: list[str], dest_dir: Path, description: str,
                   manual_instructions: str | None = None):
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if any CSV/data files already exist in dest_dir
    existing = list(dest_dir.glob("*.csv")) + list(dest_dir.glob("*.zip"))
    if existing:
        print(f"  [skip] {description} already downloaded ({len(existing)} files)")
        return existing[0]

    for url in urls:
        filename = url.split("/")[-1]
        filepath = dest_dir / filename
        print(f"  [download] {description} from {url} ...")
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  [done] Saved to {filepath}")

            if filepath.suffix == ".zip":
                print(f"  [extract] Unzipping {filename} ...")
                with zipfile.ZipFile(filepath, "r") as zf:
                    zf.extractall(dest_dir)
                print(f"  [done] Extracted to {dest_dir}")

            return filepath
        except Exception as e:
            print(f"  [error] Failed: {e}")
            if filepath.exists():
                filepath.unlink()
            continue

    print(f"  [warn] All download URLs failed for {description}")
    if manual_instructions:
        print(f"  {manual_instructions}")
    print(f"  >> Synthetic fallback data will be used during processing.")
    return None


def main():
    print("=" * 60)
    print("Fair Manufacturing Service Recommendation — Data Download")
    print("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # MSKG: manual download required
    mskg_dir = RAW_DIR / "mskg"
    if not mskg_dir.exists() or not any(mskg_dir.iterdir()):
        print(MSKG_INFO)
        mskg_dir.mkdir(parents=True, exist_ok=True)
    else:
        print("[ok] MSKG data found")

    # Automated downloads
    for name, info in DATASETS.items():
        print(f"\n--- {name} ---")
        download_file(
            info["urls"], info["dest"], info["description"],
            info.get("manual_instructions"),
        )

    print("\n" + "=" * 60)
    print("Download complete. Run preprocessing next.")
    print("=" * 60)


if __name__ == "__main__":
    main()
