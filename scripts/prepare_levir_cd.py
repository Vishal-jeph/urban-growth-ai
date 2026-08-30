"""Download and stage the LEVIR-CD building change-detection dataset.

LEVIR-CD (Chen & Shi, 2020) is free for academic use, not commercial use.
This pulls the official train/val/test splits (445 / 64 / 128 image pairs,
1024x1024, 0.5m/pixel) from a public mirror and lays them out as
data/{train,val,test}/{A,B,masks}/*.png, matching what
app.models.dataset.ChangeDetectionDataset expects.

Usage:
    python scripts/prepare_levir_cd.py [--splits train val test] [--keep-zips]
"""
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

MIRROR = "https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/main"
ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = ROOT / "data" / "_levir_cd_download"


def download(split: str) -> Path:
    zip_path = DOWNLOAD_DIR / f"{split}.zip"

    if zip_path.exists():
        print(f"[{split}] already downloaded, skipping fetch")
        return zip_path

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    url = f"{MIRROR}/{split}.zip"
    print(f"[{split}] downloading {url}")

    urllib.request.urlretrieve(url, zip_path)

    return zip_path


def stage(split: str, zip_path: Path):
    extract_dir = DOWNLOAD_DIR / f"{split}_extracted"

    print(f"[{split}] extracting {zip_path.name}")

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    dest = ROOT / "data" / split

    for src_name, dest_name in (("A", "A"), ("B", "B"), ("label", "masks")):
        src_dir = extract_dir / src_name
        dest_dir = dest / dest_name

        if dest_dir.exists():
            shutil.rmtree(dest_dir)

        shutil.move(str(src_dir), str(dest_dir))

    print(f"[{split}] staged at {dest} "
          f"({len(list((dest / 'A').glob('*'))) } pairs)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val"]
    )

    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Don't delete the downloaded zip/extract scratch dir when done."
    )

    args = parser.parse_args()

    for split in args.splits:
        zip_path = download(split)
        stage(split, zip_path)

    if not args.keep_zips:
        shutil.rmtree(DOWNLOAD_DIR)


if __name__ == "__main__":
    main()
