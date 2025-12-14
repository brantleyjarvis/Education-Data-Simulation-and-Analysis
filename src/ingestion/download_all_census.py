"""
Download all required Census/TIGER files for the project.
"""

from pathlib import Path

from src.ingestion.download_census import download_file, unzip_file


FILES = [
    {
        "url": "https://www2.census.gov/geo/tiger/TIGER2023/PLACE/tl_2023_51_place.zip",
        "out": Path("data/raw/census/tiger/2023/place"),
    },
    {
        "url": "https://www2.census.gov/geo/tiger/TIGER2023/ZCTA520/tl_2023_us_zcta520.zip",
        "out": Path("data/raw/census/tiger/2023/zcta520"),
    },
]


def main() -> None:
    for f in FILES:
        zip_path = download_file(
            url=f["url"],
            out_dir=f["out"],
        )
        unzip_file(zip_path, out_dir=f["out"])


if __name__ == "__main__":
    main()
