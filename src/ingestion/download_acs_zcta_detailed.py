from __future__ import annotations

import os
import urllib.request
import pandas as pd

ACS_URL = "https://api.census.gov/data/2023/acs/acs5"
OUT_CSV = "data/raw/acs/acs_2023_acs5_zcta_detailed.csv"

# Table variables (acs/acs5 endpoint)
VARS = [
    "B19013_001E",  # median household income
    "B25077_001E",  # median home value
    "B15003_001E",  # edu total (25+)
    "B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E",  # bachelors+
    "B09001_001E",  # children count
    "B01001_001E",  # total population
]

def _get(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    import json
    rows = json.loads(data.decode("utf-8"))
    header, values = rows[0], rows[1:]
    return pd.DataFrame(values, columns=header)

def main() -> None:
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    var_str = ",".join(["NAME"] + VARS)
    url = f"{ACS_URL}?get={var_str}&for=zip%20code%20tabulation%20area:*"
    print("[download]", url)

    df = _get(url).rename(columns={"zip code tabulation area": "zip_code"})

    df = df.rename(columns={
        "B19013_001E": "zip_income_median",
        "B25077_001E": "zip_home_value_median",
        "B09001_001E": "zip_children_count",
        "B01001_001E": "zip_population",
        "B15003_001E": "edu_total_25plus",
        "B15003_022E": "edu_bachelors",
        "B15003_023E": "edu_masters",
        "B15003_024E": "edu_professional",
        "B15003_025E": "edu_doctorate",
    })

    num_cols = [
        "zip_income_median","zip_home_value_median","zip_children_count","zip_population",
        "edu_total_25plus","edu_bachelors","edu_masters","edu_professional","edu_doctorate"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["zip_pct_bachelors_plus"] = (
        (df["edu_bachelors"] + df["edu_masters"] + df["edu_professional"] + df["edu_doctorate"])
        / df["edu_total_25plus"]
    )
    df["zip_pct_children"] = df["zip_children_count"] / df["zip_population"]

    out = df[[
        "zip_code",
        "zip_income_median",
        "zip_home_value_median",
        "zip_pct_bachelors_plus",
        "zip_pct_children",
    ]].copy()

    out.to_csv(OUT_CSV, index=False)
    print("[done] wrote:", OUT_CSV, "rows:", len(out))

if __name__ == "__main__":
    main()
