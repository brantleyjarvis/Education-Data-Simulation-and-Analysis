from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ACS_SENTINELS = [-666666666, -222222222, -999999999]


def build_city_zip_unique(city_zip_csv: Path, seed: int = 42) -> pd.DataFrame:
    """
    Read city_zip_map.csv (NAME, ZCTA5CE20) and assign each ZIP to exactly one city
    probabilistically based on frequency within ZIP.
    """
    rng = np.random.default_rng(seed)

    city_zip = pd.read_csv(city_zip_csv, dtype={"ZCTA5CE20": str})
    city_zip = city_zip.rename(columns={"NAME": "city", "ZCTA5CE20": "zip_code"})

    freq = (
        city_zip.groupby(["zip_code", "city"])
        .size()
        .reset_index(name="count")
    )

    freq["prob"] = freq.groupby("zip_code")["count"].transform(lambda x: x / x.sum())

    def pick_city(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "zip_code": group["zip_code"].iloc[0],
                "city": rng.choice(group["city"].values, p=group["prob"].values),
            }
        )

    city_zip_unique = (
        freq.groupby("zip_code", as_index=False)
        .apply(pick_city)
        .reset_index(drop=True)
    )

    city_zip_unique["zip_code"] = city_zip_unique["zip_code"].astype(str).str.zfill(5)
    return city_zip_unique


def build_zip_df(
    city_zip_csv: Path,
    acs_detailed_csv: Path,
    acs_profile_csv: Path,
    target_city_weights: dict[str, float],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build the ZIP-level dataframe used by the simulator by merging:
      - city_zip_map.csv (processed output)
      - ACS detailed + profile CSVs (user-provided)
    """
    city_zip_unique = build_city_zip_unique(city_zip_csv=city_zip_csv, seed=seed)

    acs_detailed_all = pd.read_csv(acs_detailed_csv, dtype={"zip_code": str})
    acs_profile_all = pd.read_csv(acs_profile_csv, dtype={"zip_code": str})

    zips = city_zip_unique["zip_code"].tolist()

    acs_detailed = acs_detailed_all[acs_detailed_all["zip_code"].isin(zips)].copy()
    acs_profile = acs_profile_all[acs_profile_all["zip_code"].isin(zips)].copy()

    zip_acs = acs_detailed.merge(
        acs_profile.drop(columns=["NAME"], errors="ignore"),
        on="zip_code",
        how="left",
    )

    zip_acs = zip_acs.rename(
        columns={
            "B19013_001E": "zip_income_median",
            "B25077_001E": "zip_home_value_median",
            "DP02_0068PE": "zip_pct_bachelors_plus",
            "DP05_0019PE": "zip_pct_children",
        }
    )

    num_cols = [
        "zip_income_median",
        "zip_home_value_median",
        "zip_pct_bachelors_plus",
        "zip_pct_children",
    ]
    for col in num_cols:
        zip_acs[col] = pd.to_numeric(zip_acs[col], errors="coerce").replace(ACS_SENTINELS, np.nan)

    city_zip_unique["zip_code"] = city_zip_unique["zip_code"].astype(str)
    zip_acs["zip_code"] = zip_acs["zip_code"].astype(str)

    zip_df = city_zip_unique.merge(
        zip_acs.drop(columns=["NAME"], errors="ignore"),
        on="zip_code",
        how="left",
    )

    ses_cols = ["zip_income_median", "zip_home_value_median", "zip_pct_bachelors_plus"]
    zip_df[ses_cols] = zip_df[ses_cols].apply(lambda s: s.fillna(s.mean()))

    for col in ses_cols:
        zip_df[col + "_z"] = (zip_df[col] - zip_df[col].mean()) / zip_df[col].std()

    zip_df["ses_index"] = (
        zip_df["zip_income_median_z"]
        + zip_df["zip_home_value_median_z"]
        + zip_df["zip_pct_bachelors_plus_z"]
    )

    zip_df["city_weight"] = zip_df["city"].map(target_city_weights).fillna(0.0)
    zip_counts = zip_df.groupby("city")["zip_code"].transform("count")
    zip_df["sampling_weight"] = zip_df["city_weight"] / zip_counts
    zip_df["sampling_weight"] = zip_df["sampling_weight"] / zip_df["sampling_weight"].sum()

    return zip_df
