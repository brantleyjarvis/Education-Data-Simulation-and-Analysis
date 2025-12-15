from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_applicants(
    n_applicants: int,
    zip_df: pd.DataFrame,
    rng: np.random.Generator | None = None
) -> pd.DataFrame:
    """
    Simulate n_applicants using the private school generative model.

    Expected columns in zip_df:
      - sampling_weight
      - zip_code, city
      - zip_income_median, zip_home_value_median
      - zip_pct_bachelors_plus, zip_pct_children
      - ses_index
    """
    if rng is None:
        rng = np.random.default_rng()

    df = pd.DataFrame()

    # 1. Applicant ID
    df["applicant_id"] = np.arange(1, n_applicants + 1)

    # 2. Grade applying to
    grades = np.arange(1, 13)
    grade_probs = np.array([
        0.25, 0.08, 0.09, 0.07, 0.07, 0.15,
        0.06, 0.05, 0.10, 0.04, 0.02, 0.02
    ])
    df["grade_applying_to"] = rng.choice(grades, size=n_applicants, p=grade_probs)

    # 3. Tuition by grade
    tuition_by_grade = {
        1: 23700, 2: 23700, 3: 23700,
        4: 25000, 5: 25000, 6: 25000,
        7: 28100, 8: 28100, 9: 28100,
        10: 29400, 11: 29400, 12: 29400
    }
    df["tuition"] = df["grade_applying_to"].map(tuition_by_grade)

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

    # 5. Race / Ethnicity
    def baseline_minority_prob(g: int) -> float:
        if g <= 5:
            return 0.26
        elif g <= 8:
            return 0.29
        else:
            return 0.24

    baseline_probs = df["grade_applying_to"].apply(baseline_minority_prob)
    ses_centered = df["ses_index"] - df["ses_index"].mean()
    df["ses_centered"] = ses_centered

    logit = np.log(baseline_probs / (1 - baseline_probs)) - 0.10 * ses_centered
    minority_prob = 1 / (1 + np.exp(-logit))
    df["race_ethnic_minority"] = rng.binomial(1, minority_prob)

    # 6. Gender
    df["gender_male"] = rng.binomial(1, 0.5, size=len(df))

    # 7. Income band
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
        size=len(df),
        p=[0.05, 0.25, 0.45, 0.20, 0.05]
    )

    # 9. Tuition-enrolled children
    base_prob = df["family_size"].map({2: 0.0, 3: 0.05, 4: 0.15, 5: 0.25, 6: 0.35})
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

    legacy_prob = 1 / (1 + np.exp(-logit))
    df["legacy_status"] = rng.binomial(1, legacy_prob)

    # 11. Aid requested
base_p = df["income_band"].map({
    "<75k": 0.95,      # was 0.98
    "75–150k": 0.85,   # was 0.95
    "150–250k": 0.20,  # was 0.30
    ">250k": 0.03      # was 0.05
})

logit = np.log(base_p / (1 - base_p))
logit += 0.05 * (df["family_size"] - 3)

# soften the "already paying tuition" push a bit
logit += 0.05 * df["tuition_enrolled_children"]   # was 0.10

# stronger SES deterrent
logit -= 0.10 * df["ses_centered"]                # was 0.05

aid_prob = 1 / (1 + np.exp(-logit))
df["aid_requested"] = rng.binomial(1, aid_prob)


    # 12. Scores
    n = len(df)

    corr = np.array([
        [1.0, 0.5, 0.4, 0.5, 0.5],
        [0.5, 1.0, 0.3, 0.4, 0.4],
        [0.4, 0.3, 1.0, 0.3, 0.3],
        [0.5, 0.4, 0.3, 1.0, 0.5],
        [0.5, 0.4, 0.3, 0.5, 1.0],
    ])

    L = np.linalg.cholesky(corr)
    latent = rng.normal(size=(n, 5)) @ L.T

    z_testing, z_art, z_ath, z_lead, z_interview = latent.T

    ses = df["ses_centered"].to_numpy()
    minority = df["race_ethnic_minority"].to_numpy()

    z_testing += 0.20 * ses - 0.20 * minority
    z_art += 0.10 * ses - 0.05 * minority
    z_ath += 0.10 * ses - 0.05 * minority
    z_lead += 0.10 * ses - 0.05 * minority
    z_interview += 0.10 * ses - 0.05 * minority

    rank_pct = pd.Series(z_testing).rank(pct=True)
    df["score_testing"] = (rank_pct ** 0.4 * 100).clip(0, 99)

    def likert(z: np.ndarray, cuts: tuple[float, float, float, float]) -> np.ndarray:
        pct = pd.Series(z).rank(pct=True)
        return np.select(
            [pct <= cuts[0], pct <= cuts[1], pct <= cuts[2], pct <= cuts[3]],
            [1, 2, 3, 4],
            default=5
        )

    df["score_art"] = likert(z_art + rng.normal(0, 0.35, n), (0.12, 0.32, 0.58, 0.82))
    df["score_athletics"] = likert(z_ath + rng.normal(0, 0.35, n), (0.08, 0.28, 0.55, 0.80))
    df["score_leadership"] = likert(z_lead + rng.normal(0, 0.35, n), (0.10, 0.30, 0.55, 0.78))
    df["score_interview"] = likert(z_interview + rng.normal(0, 0.35, n), (0.06, 0.25, 0.50, 0.75))

    # 13. spot_offered (admissions)
    seat_quota = {
        1: 40, 2: 15, 3: 15, 4: 12, 5: 12,
        6: 30, 7: 8, 8: 8, 9: 20,
        10: 3, 11: 3, 12: 4
    }

    df["spot_offered"] = 0

    z_test = (df["score_testing"] - df["score_testing"].mean()) / df["score_testing"].std()
    z_int = (df["score_interview"] - df["score_interview"].mean()) / df["score_interview"].std()
    z_lead_score = (df["score_leadership"] - df["score_leadership"].mean()) / df["score_leadership"].std()

    income_bonus = df["income_band"].map({
        "<75k": 0.6, "75–150k": 0.3, "150–250k": 0.1, ">250k": 0.0
    })

    admit_index = (
        0.6 * z_test +
        0.3 * z_int +
        0.3 * z_lead_score +
        0.6 * df["legacy_status"] +
        income_bonus
    )

    df["admit_index"] = admit_index

    for g, n_seats in seat_quota.items():
        rows = df[df["grade_applying_to"] == g]
        if len(rows) == 0:
            continue
        admit = rows.sort_values("admit_index", ascending=False).head(n_seats).index
        df.loc[admit, "spot_offered"] = 1

    # 14. Aid offered
df["aid_offered_pct_tuition"] = 0.0
df["aid_offered_amount"] = 0.0

eligible = (df["spot_offered"] == 1) & (df["aid_requested"] == 1)
sub = df.loc[eligible].copy()

# ~85% of aid applicants receive aid (award gate) ---
AID_AWARD_RATE = 0.85
sub["aid_awarded"] = rng.binomial(1, AID_AWARD_RATE, size=len(sub)).astype(bool)

# only compute award amounts for those awarded
sub_awarded = sub.loc[sub["aid_awarded"]].copy()

band_mean = {
    "<75k": 0.80, "75–150k": 0.55,
    "150–250k": 0.25, ">250k": 0.05
}

pct = (
    sub_awarded["income_band"].map(band_mean)
    + 0.04 * (sub_awarded["family_size"] - 4)
    - 0.03 * sub_awarded["ses_centered"]
    + 0.03 * sub_awarded["race_ethnic_minority"]
    + 0.03 * sub_awarded["tuition_enrolled_children"]
    + rng.normal(0, 0.08, len(sub_awarded))
).clip(0, 1)

# don't count tiny token awards as "receiving aid"
MIN_AID_PCT = 0.05
pct = np.where(pct < MIN_AID_PCT, 0.0, pct)

df.loc[sub_awarded.index, "aid_offered_pct_tuition"] = pct
df.loc[sub_awarded.index, "aid_offered_amount"] = pct * sub_awarded["tuition"]

    # 15. Enrollment decision
    df["enrolled"] = 0
    mask = df["spot_offered"] == 1

    aid_slope = df.loc[mask, "income_band"].map({
        "<75k": 3.5, "75–150k": 2.8,
        "150–250k": 1.6, ">250k": 0.8
    })

    z_test_masked = (df.loc[mask, "score_testing"] - df["score_testing"].mean()) / df["score_testing"].std()

    grade_effect = df.loc[mask, "grade_applying_to"].apply(
        lambda g: 0.0 if g <= 5 else 0.3 if g <= 8 else 0.6
    )

    logit = (
        1.0
        + 0.8 * df.loc[mask, "aid_offered_pct_tuition"]
        + aid_slope * df.loc[mask, "aid_offered_pct_tuition"]
        + 0.9 * df.loc[mask, "legacy_status"]
        + 0.6 * df.loc[mask, "tuition_enrolled_children"]
        + 0.25 * z_test_masked
        + grade_effect
    )

    enroll_prob = 1 / (1 + np.exp(-logit))
    df.loc[mask, "enrolled"] = rng.binomial(1, enroll_prob)

    return df


def simulate_many_years(
    n_years: int,
    applicants_per_year: int,
    zip_df: pd.DataFrame,
    base_seed: int = 42
) -> pd.DataFrame:
    """Simulate multiple years of applicants by varying RNG seed by year."""
    all_years: list[pd.DataFrame] = []
    for year in range(1, n_years + 1):
        rng_year = np.random.default_rng(base_seed + year)
        df_year = simulate_applicants(applicants_per_year, zip_df, rng_year)
        df_year["year"] = year
        all_years.append(df_year)
    return pd.concat(all_years, ignore_index=True)
