"""
Download Census/TIGER files into the repo's data/ directory.

Example:
  python -m src.ingestion.download_census \
    --url https://www2.census.gov/geo/tiger/TIGER2023/PLACE/tl_2023_51_place.zip \
    --out data/raw/census/tiger/2023/place \
    --unzip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests


def _filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name
    if not name:
        raise ValueError(f"Could not infer filename from url: {url}")
    return name


def download_file(
    url: str,
    out_dir: Path,
    filename: str | None = None,
    overwrite: bool = False,
    timeout: int = 60,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = _filename_from_url(url)

    out_path = out_dir / filename

    if out_path.exists() and not overwrite:
        print(f"[skip] already exists: {out_path}")
        return out_path

    print(f"[download] {url}")
    print(f"          -> {out_path}")

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()

        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        with tmp_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        tmp_path.replace(out_path)

    print(f"[done] downloaded: {out_path}")
    return out_path


def unzip_file(zip_path: Path, out_dir: Path, overwrite: bool = False) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    unzip_dir = out_dir / zip_path.stem
    unzip_dir.mkdir(parents=True, exist_ok=True)

    print(f"[unzip] {zip_path} -> {unzip_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            target = unzip_dir / member
            if target.exists() and not overwrite:
                continue
            z.extract(member, path=unzip_dir)

    print(f"[done] unzipped to: {unzip_dir}")
    return unzip_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Census/TIGER files.")
    parser.add_argument("--url", required=True, help="Direct URL to download.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--filename", default=None, help="Optional override filename.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite if file exists.")
    parser.add_argument("--unzip", action="store_true", help="Unzip after download.")
    args = parser.parse_args(argv)

    try:
        out_dir = Path(args.out).expanduser().resolve()
        zip_path = download_file(
            url=args.url,
            out_dir=out_dir,
            filename=args.filename,
            overwrite=args.overwrite,
        )

        if args.unzip and zip_path.suffix.lower() == ".zip":
            unzip_file(zip_path, out_dir=out_dir)

        return 0

    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
