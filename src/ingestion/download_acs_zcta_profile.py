from __future__ import annotations

import os
import urllib.request
import pandas as pd

ACS_PROFILE_URL = "https://api.census.gov/data/2023/acs/acs5/profile"
OUT_CSV = "data/raw/acs/acs_2023_acs5_zcta_profile.csv"

VARS = [
    "DP05_0001E",   # total population
    "DP03_0062E",   # median household income
    "DP04_0089E",   # median home value
    "DP02_0068PE",  # % bachelor's degree or higher
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
    url = f"{ACS_PROFILE_URL}?get={var_str}&for=zip%20code%20tabulation%20area:*"
    print("[download]", url)

    df = _get(url).rename(columns={"zip code tabulation area": "zip_code"})
    df = df.rename(columns={
        "DP05_0001E": "zip_population",
        "DP03_0062E": "zip_income_median",
        "DP04_0089E": "zip_home_value_median",
        "DP02_0068PE": "zip_pct_bachelors_plus",
    })

    for c in ["zip_population","zip_income_median","zip_home_value_median","zip_pct_bachelors_plus"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = df[["zip_code","zip_population","zip_income_median","zip_home_value_median","zip_pct_bachelors_plus"]].copy()
    out.to_csv(OUT_CSV, index=False)
    print("[done] wrote:", OUT_CSV, "rows:", len(out))

if __name__ == "__main__":
    main()
