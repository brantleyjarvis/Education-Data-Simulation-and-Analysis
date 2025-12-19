from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.ingestion.build_zip_df import build_zip_df
from src.simulation.finaid_sim import simulate_many_years_from_targets


def main() -> None:
    city_zip_csv = Path("data/processed/geo/city_zip_map.csv")

    # Place your ACS exports here (these should be gitignored):
    acs_detailed_csv = Path("data/raw/acs/acs_2023_acs5_zcta_detailed.csv")
    acs_profile_csv = Path("data/raw/acs/acs_2023_acs5_zcta_profile.csv")

    target_city_weights = {
        "Virginia Beach": 0.63,
        "Norfolk": 0.21,
        "Chesapeake": 0.10,
        "Suffolk": 0.02,
        "Portsmouth": 0.02,
    }

    zip_df = build_zip_df(
        city_zip_csv=city_zip_csv,
        acs_detailed_csv=acs_detailed_csv,
        acs_profile_csv=acs_profile_csv,
        target_city_weights=target_city_weights,
        seed=42,
    )

    # -----------------------------
    # TARGETS-DRIVEN SIMULATION
    # -----------------------------
    # Use the bootstrapped/extended targets you created.
    targets_path = Path("data/private/grade_year_targets_bootstrap.csv")  # change if needed
    targets = pd.read_csv(targets_path)

    # Respect n_years by subsetting to first N unique years in targets
    n_years = 10
    years = sorted(targets["year"].unique())
    keep_years = years[:n_years]
    targets = targets[targets["year"].isin(keep_years)].copy()

    sim_df = simulate_many_years_from_targets(
        targets=targets,
        zip_df=zip_df,
        base_seed=42,
    )

    sim_df["data_type"] = "simulated"

    out_dir = Path("data/processed/sim")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "finaid_sim.csv"
    sim_df.to_csv(out_path, index=False)

    print(f"Saved simulation to: {out_path}")
    print(f"Used targets: {targets_path}")
    print(f"Years simulated: {sorted(sim_df['year'].unique())}")


if __name__ == "__main__":
    main()
