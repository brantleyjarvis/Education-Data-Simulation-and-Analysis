"""
TARGETS-DRIVEN SIMULATION (DROP-IN BLOCK)

What this does (in plain English):
- Uses Brian's targets table (grade x year) to drive HOW MANY applicants are created per grade-year.
- Generates applicant characteristics + scores (Steps 1–12).
- Then enforces offers + enrollment totals from the targets table (instead of your internal seat/yield logic).
- Keeps the microdata realistic by choosing who gets offered via admit_index and who enrolls via an enrollment-probability model.
- DOES NOT run your old admissions/waitlist/aid pipeline inside simulate_applicants (that would conflict with targets).
- Leaves aid for a later step (once offers/enrollments match targets). You can add it back after this runs cleanly.

Expected targets columns:
- year (int)
- grade (int 1–12)
- apps (int)  [required]
AND EITHER:
- offers (int) and enrolled (int)
OR:
- offers (int) and yield (float 0–1)
OR:
- seats (int) and yield (float 0–1)

If your table uses different names, adjust the small COLUMN MAP section below.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# GLOBAL CONSTANTS
# -----------------------------
TUITION_BY_GRADE = {
    1: 23700, 2: 23700, 3: 23700,
    4: 25000, 5: 25000, 6: 25000,
    7: 28100, 8: 28100, 9: 28100,
    10: 29400, 11: 29400, 12: 29400
}

# -----------------------------
# 1) Applicant generator (Steps 1–12 only)
# -----------------------------
def simulate_applicants(
    n_applicants: int,
    zip_df: pd.DataFrame,
    rng: np.random.Generator | None = None,
    forced_grade: int | None = None,
) -> pd.DataFrame:
    """
    Generate applicant microdata up through scores.
    Admissions/offers/enrollment will be enforced from targets elsewhere.
    """
    if rng is None:
        rng = np.random.default_rng()

    df = pd.DataFrame()

    # 1. Applicant ID
    df["applicant_id"] = np.arange(1, n_applicants + 1)

    # 2. Grade applying to
    if forced_grade is not None:
        df["grade_applying_to"] = forced_grade
    else:
        grades = np.arange(1, 13)
        grade_probs = np.array([
            0.25, 0.08, 0.09, 0.07, 0.07, 0.15,
            0.06, 0.05, 0.10, 0.04, 0.02, 0.02
        ])
        df["grade_applying_to"] = rng.choice(grades, size=n_applicants, p=grade_probs)

    # 3. Tuition by grade
    df["tuition"] = df["grade_applying_to"].map(TUITION_BY_GRADE)

    # 4. ZIP assignment + SES
    zip_idx = rng.choice(
        zip_df.index.to_numpy(),
        size=n_applicants,
        p=zip_df["sampling_weight"].to_numpy()
    )
    chosen_zips = zip_df.loc[zip_idx].reset_index(drop=True)

    cols_to_add = [
        "zip_code", "city",
        "zip_income_median", "zip_home_value_median",
        "zip_pct_bachelors_plus", "zip_pct_children",
        "ses_index"
    ]
    df = pd.concat([df, chosen_zips[cols_to_add]], axis=1)
    df["ses_index"] = df["ses_index"].fillna(df["ses_index"].mean())
    df["ses_centered"] = df["ses_index"] - df["ses_index"].mean()

    # 5. Race / Ethnicity
    def baseline_minority_prob(g: int) -> float:
        if g <= 5:
            return 0.26
        elif g <= 8:
            return 0.29
        else:
            return 0.24

    base_minority = df["grade_applying_to"].apply(baseline_minority_prob)
    logit = np.log(base_minority / (1 - base_minority)) - 0.10 * df["ses_centered"]
    minority_prob = 1 / (1 + np.exp(-logit))
    df["race_ethnic_minority"] = rng.binomial(1, minority_prob)

    # 6. Gender
    df["gender_male"] = rng.binomial(1, 0.5, size=n_applicants)

    # 7. Income band (driven by SES percentile)
    ses_pct = df["ses_index"].rank(pct=True)

    def assign_income_band(p: float) -> str:
        if p < 0.10:
            return rng.choice(["<75k", "75–150k"], p=[0.7, 0.3])
        elif p < 0.30:
            return rng.choice(["75–150k", "150–250k"], p=[0.6, 0.4])
        elif p < 0.70:
            return rng.choice(["150–250k", ">250k"], p=[0.6, 0.4])
        else:
            return rng.choice([">250k", "150–250k"], p=[0.7, 0.3])

    df["income_band"] = ses_pct.apply(assign_income_band)

    # 8. Family size
    df["family_size"] = rng.choice(
        [2, 3, 4, 5, 6],
        size=n_applicants,
        p=[0.05, 0.25, 0.45, 0.20, 0.05]
    )

    # 9. Tuition-enrolled children (binary)
    base_prob = df["family_size"].map({2: 0.0, 3: 0.10, 4: 0.40, 5: 0.55, 6: 0.70})
    logit = np.log(np.clip(base_prob, 1e-6, 1 - 1e-6) / (1 - base_prob))
    logit += 0.05 * (ses_pct - 0.5)
    sib_prob = 1 / (1 + np.exp(-logit))
    df["tuition_enrolled_children"] = rng.binomial(1, sib_prob)
    df.loc[df["family_size"] == 2, "tuition_enrolled_children"] = 0

    # 10. Legacy status
    city_base = df["city"].map({
        "Virginia Beach": 0.28,
        "Norfolk": 0.24,
        "Chesapeake": 0.22,
        "Suffolk": 0.20,
        "Portsmouth": 0.20
    }).fillna(0.22)

    def grade_legacy_effect(g: int) -> float:
        if g <= 3:
            return 0.3
        elif g <= 8:
            return 0.0
        else:
            return -0.3

    logit = np.log(city_base / (1 - city_base))
    logit += df["grade_applying_to"].apply(grade_legacy_effect)
    logit += 0.12 * df["ses_centered"]
    logit += 0.6 * df["tuition_enrolled_children"]
    df["legacy_status"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    # 11. Aid requested
    base_p = df["income_band"].map({
        "<75k": 0.99,
        "75–150k": 0.90,
        "150–250k": 0.01,
        ">250k": 0.0001
    })
    logit = np.log(base_p / (1 - base_p))
    logit += 0.05 * (df["family_size"] - 3)
    logit += 0.05 * df["tuition_enrolled_children"]
    logit -= 0.10 * df["ses_centered"]
    df["aid_requested"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    # 12. Scores
    n = len(df)
    corr = np.array([
        [1.0, 0.5, 0.4, 0.5, 0.5],
        [0.5, 1.0, 0.3, 0.4, 0.4],
        [0.4, 0.3, 1.0, 0.3, 0.3],
        [0.5, 0.4, 0.3, 1.0, 0.5],
        [0.5, 0.4, 0.3, 0.5, 1.0],
    ])
    latent = rng.normal(size=(n, 5)) @ np.linalg.cholesky(corr).T
    z_test, z_art, z_ath, z_lead, z_int = latent.T

    z_test += 0.20 * df["ses_centered"] - 0.20 * df["race_ethnic_minority"]
    z_art  += 0.10 * df["ses_centered"] - 0.05 * df["race_ethnic_minority"]
    z_ath  += 0.10 * df["ses_centered"] - 0.05 * df["race_ethnic_minority"]
    z_lead += 0.10 * df["ses_centered"] - 0.05 * df["race_ethnic_minority"]
    z_int  += 0.10 * df["ses_centered"] - 0.05 * df["race_ethnic_minority"]

    pct = pd.Series(z_test).rank(pct=True)
    df["score_testing"] = (pct ** 0.4 * 100).clip(0, 99)

    def likert(z, cuts):
        p = pd.Series(z).rank(pct=True)
        return np.select(
            [p <= cuts[0], p <= cuts[1], p <= cuts[2], p <= cuts[3]],
            [1, 2, 3, 4],
            default=5
        )

    df["score_art"]        = likert(z_art  + rng.normal(0, 0.35, n), (0.12, 0.32, 0.58, 0.82))
    df["score_athletics"]  = likert(z_ath  + rng.normal(0, 0.35, n), (0.08, 0.28, 0.55, 0.80))
    df["score_leadership"] = likert(z_lead + rng.normal(0, 0.35, n), (0.10, 0.30, 0.55, 0.78))
    df["score_interview"]  = likert(z_int  + rng.normal(0, 0.35, n), (0.06, 0.25, 0.50, 0.75))

    return df

def simulate_many_years_from_targets(
    targets: pd.DataFrame,
    zip_df: pd.DataFrame,
    base_seed: int = 42
) -> pd.DataFrame:
    all_years = []

    for year in sorted(targets["year"].unique()):
        rng_year = np.random.default_rng(base_seed + int(year))

        # Simulate applicants grade-by-grade using observed Apps
        df_year_parts = []
        for g in range(1, 13):
            row = targets[(targets["year"] == year) & (targets["grade"] == g)]
            if row.empty:
                continue

            n_apps = int(row["apps"].iloc[0])
            df_g = simulate_applicants(n_apps, zip_df, rng_year, forced_grade=g)
            df_g["year"] = year
            df_year_parts.append(df_g)

        df_year = pd.concat(df_year_parts, ignore_index=True)

        # Apply observed offers/enrollment calibration
        df_year = apply_offers_and_enrollment_from_targets(df_year, targets, int(year), rng_year)

        all_years.append(df_year)

    return pd.concat(all_years, ignore_index=True)

def simulate_many_years(
    n_years: int,
    zip_df: pd.DataFrame,
    rng: np.random.Generator | None = None,
    base_seed: int = 42,
    targets: pd.DataFrame | None = None,
    targets_path: str = "data/private/grade_year_targets_bootstrap.csv",
) -> pd.DataFrame:
    """
    Backwards-compatible entry point expected by run_simulation.py.

    Old behavior: simulate N years without targets.
    New behavior (default): simulate using Brian targets (bootstrapped/extended table).

    - If `targets` is None, loads from `targets_path`.
    - Ignores `n_years` if targets already contain the year range; otherwise truncates/extends.
    """
    if rng is None:
        rng = np.random.default_rng(base_seed)

    if targets is None:
        targets = pd.read_csv(targets_path)

    # If caller requested n_years, subset to that many unique years (stable order)
    years = sorted(targets["year"].unique())
    if n_years is not None and len(years) > n_years:
        keep_years = years[:n_years]
        targets = targets[targets["year"].isin(keep_years)].copy()

    return simulate_many_years_from_targets(targets=targets, zip_df=zip_df, base_seed=base_seed)

