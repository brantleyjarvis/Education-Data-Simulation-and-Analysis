from __future__ import annotations

from pathlib import Path
import geopandas as gpd


DEFAULT_TARGET_CITIES = [
    "Portsmouth",
    "Norfolk",
    "Virginia Beach",
    "Suffolk",
    "Chesapeake",
]


def build_city_zip_map(
    place_zip: Path,
    zcta_zip: Path,
    target_cities: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Load TIGER PLACE + ZCTA files (from ZIPs) and return ZCTA polygons that
    intersect the selected cities, with city name and ZCTA code columns.

    Returns a GeoDataFrame with at least:
      - NAME (city name from PLACE)
      - ZCTA5CE20 (ZCTA code)
      - geometry (ZCTA geometry)
    """
    if target_cities is None:
        target_cities = DEFAULT_TARGET_CITIES

    if not place_zip.exists():
        raise FileNotFoundError(f"PLACE zip not found: {place_zip}")
    if not zcta_zip.exists():
        raise FileNotFoundError(f"ZCTA zip not found: {zcta_zip}")

    place = gpd.read_file(f"zip://{place_zip}!tl_2023_51_place.shp")
    zcta = gpd.read_file(f"zip://{zcta_zip}!tl_2023_us_zcta520.shp")

    cities = place[place["NAME"].str.lower().isin([c.lower() for c in target_cities])]

    zcta = zcta.to_crs(cities.crs)

    city_zips = gpd.sjoin(zcta, cities, how="inner", predicate="intersects")
    return city_zips


def save_city_zip_outputs(city_zips: gpd.GeoDataFrame, out_dir: Path) -> tuple[Path, Path]:
    """
    Save a tidy CSV (NAME, ZCTA5CE20) and a GeoJSON with geometry.
    Returns (csv_path, geojson_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "city_zip_map.csv"
    geojson_path = out_dir / "city_zip_map.geojson"

    (
        city_zips[["NAME", "ZCTA5CE20"]]
        .drop_duplicates()
        .sort_values(["NAME", "ZCTA5CE20"])
        .to_csv(csv_path, index=False)
    )

    city_zips.to_file(geojson_path, driver="GeoJSON")

    return csv_path, geojson_path


def main() -> None:
    data_dir = Path("data/raw/census/tiger/2023")
    place_zip = data_dir / "place" / "tl_2023_51_place.zip"
    zcta_zip = data_dir / "zcta520" / "tl_2023_us_zcta520.zip"

    city_zips = build_city_zip_map(place_zip=place_zip, zcta_zip=zcta_zip)

    out_dir = Path("data/processed/geo")
    csv_path, geojson_path = save_city_zip_outputs(city_zips, out_dir=out_dir)

    print("Shapefiles loaded and spatial join complete.")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved GeoJSON: {geojson_path}")


if __name__ == "__main__":
    main()
