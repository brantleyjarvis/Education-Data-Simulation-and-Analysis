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
        size=n_applicants,
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
    df["legacy_status"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    # 11. Aid requested
    base_p = df["income_band"].map({
        "<75k": 0.95,
        "75–150k": 0.85,
        "150–250k": 0.20,
        ">250k": 0.03
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
    z_art += 0.10 * df["ses_centered"] - 0.05 * df["race_ethnic_minority"]
    z_ath += 0.10 * df["ses_centered"] - 0.05 * df["race_ethnic_minority"]
    z_lead += 0.10 * df["ses_centered"] - 0.05 * df["race_ethnic_minority"]
    z_int += 0.10 * df["ses_centered"] - 0.05 * df["race_ethnic_minority"]

    pct = pd.Series(z_test).rank(pct=True)
    df["score_testing"] = (pct ** 0.4 * 100).clip(0, 99)

    def likert(z, cuts):
        p = pd.Series(z).rank(pct=True)
        return np.select(
            [p <= cuts[0], p <= cuts[1], p <= cuts[2], p <= cuts[3]],
            [1, 2, 3, 4],
            default=5
        )

    df["score_art"] = likert(z_art + rng.normal(0, 0.35, n), (0.12, 0.32, 0.58, 0.82))
    df["score_athletics"] = likert(z_ath + rng.normal(0, 0.35, n), (0.08, 0.28, 0.55, 0.80))
    df["score_leadership"] = likert(z_lead + rng.normal(0, 0.35, n), (0.10, 0.30, 0.55, 0.78))
    df["score_interview"] = likert(z_int + rng.normal(0, 0.35, n), (0.06, 0.25, 0.50, 0.75))

    # 13. Spot offered
    seat_quota = {
        1: 40, 2: 15, 3: 15, 4: 12, 5: 12,
        6: 30, 7: 8, 8: 8, 9: 20,
        10: 3, 11: 3, 12: 4
    }

    df["spot_offered"] = 0

    z_test = (df["score_testing"] - df["score_testing"].mean()) / df["score_testing"].std()
    z_int = (df["score_interview"] - df["score_interview"].mean()) / df["score_interview"].std()
    z_lead = (df["score_leadership"] - df["score_leadership"].mean()) / df["score_leadership"].std()

    income_bonus = df["income_band"].map({
        "<75k": 0.6, "75–150k": 0.3, "150–250k": 0.1, ">250k": 0.0
    })

    df["admit_index"] = (
        0.6 * z_test +
        0.3 * z_int +
        0.3 * z_lead +
        0.6 * df["legacy_status"] +
        income_bonus
    )

    for g, seats in seat_quota.items():
        idx = df[df["grade_applying_to"] == g].sort_values(
            "admit_index", ascending=False
        ).head(seats).index
        df.loc[idx, "spot_offered"] = 1

    # 14. Aid OFFER (used to drive enrollment decision)
    df["aid_offer_pct_tuition"] = 0.0
    df["aid_offer_amount"] = 0.0

    eligible = df[(df["spot_offered"] == 1) & (df["aid_requested"] == 1)].copy()

    def rtrunc_lognormal(rng, mean, low, high, size, sigma=0.65):
        """
        Right-skewed truncated lognormal sampler (no per-draw rescaling).
        - sigma controls skew (0.5–0.8 reasonable)
        - mean is used to set the underlying lognormal mean before truncation
        """
        # guard against impossible/inconsistent params (like mean < low)
        mean_eff = min(max(mean, low * 1.001), high * 0.999)

        mu = np.log(mean_eff) - 0.5 * sigma**2

        out = np.empty(size, dtype=float)
        i = 0
        while i < size:
            draw = rng.lognormal(mean=mu, sigma=sigma, size=(size - i) * 10)
            draw = draw[(draw >= low) & (draw <= high)]
            take = min(len(draw), size - i)
            if take > 0:
                out[i:i + take] = draw[:take]
                i += take 
        return out

    def map_to_school_income_bin(income_band: str) -> str:
        if income_band == "<75k":
            return "50-100"   # approx (you don't have numeric income yet)
        if income_band == "75–150k":
            return "100-150"
        if income_band in ("150–250k", ">250k"):
            return ">150"
        return ">150"

    eligible["school_income_bin"] = eligible["income_band"].astype(str).map(map_to_school_income_bin)
    eligible["multi_child"] = (eligible["tuition_enrolled_children"] >= 2)

    ONE = {
        "<=50":    dict(p_award=29/29, mean=22474, low=10000, high=26500),
        "50-100":  dict(p_award=31/33, mean=17258, low=4750,  high=26750),
        "100-150": dict(p_award=7/13,  mean=7327,  low=8000,  high=23250),
        ">150":    dict(p_award=0/8,   mean=0,     low=0,     high=0),
    }

    MULTI = {
        "<=50":    dict(p_award=31/31, mean=24137, low=20000, high=31500),
        "50-100":  dict(p_award=41/41, mean=19902, low=10250, high=26750),
        "100-150": dict(p_award=43/43, mean=14494, low=1000,  high=28000),
        ">150":    dict(p_award=0.0,   mean=0,     low=0,     high=0),
    }

    def to_key(bin_str: str) -> str:
        if bin_str == "50-100":
            return "50-100"
        if bin_str == "100-150":
            return "100-150"
        if bin_str == ">150":
            return ">150"
        return "50-100"

    eligible["k"] = eligible["school_income_bin"].map(to_key)

    p = np.array([
        (MULTI if mc else ONE)[k]["p_award"]
        for mc, k in zip(eligible["multi_child"].to_numpy(), eligible["k"].to_numpy())
    ], dtype=float)

    eligible["aid_awarded"] = rng.random(len(eligible)) < p
    awarded = eligible[eligible["aid_awarded"]].copy()

    params = []
    for mc, k in zip(awarded["multi_child"].to_numpy(), awarded["k"].to_numpy()):
        d = (MULTI if mc else ONE)[k]
        params.append((d["mean"], max((d["high"] - d["low"]) / 6, 1.0), d["low"], d["high"]))

    means = np.array([t[0] for t in params], float)
    sds   = np.array([t[1] for t in params], float)
    lows  = np.array([t[2] for t in params], float)
    highs = np.array([t[3] for t in params], float)

    offer_amt = np.empty(len(awarded), dtype=float)
    for i in range(len(awarded)):
        if highs[i] <= 0:
            offer_amt[i] = 0.0
        else:
            offer_amt[i] = rtrunc_lognormal(
            rng,
            mean=means[i],
            low=lows[i],
            high=highs[i],
            size=1,
            sigma=0.65
        )[0]

    tuition = awarded["tuition"].to_numpy(dtype=float)
    offer_amt = np.minimum(offer_amt, tuition)

    df.loc[awarded.index, "aid_offer_amount"] = offer_amt
    df.loc[awarded.index, "aid_offer_pct_tuition"] = (offer_amt / tuition).clip(0, 1)

    # initialize final aid columns (will be overwritten post-enrollment)
    df["aid_offered_amount"] = 0.0
    df["aid_offered_pct_tuition"] = 0.0

    # 15. Enrollment decision
    df["enrolled"] = 0
    mask = df["spot_offered"] == 1

    aid_slope = df.loc[mask, "income_band"].map({
        "<75k": 3.5, "75–150k": 2.8,
        "150–250k": 1.6, ">250k": 0.8
    })

    z_test_m = (df.loc[mask, "score_testing"] - df["score_testing"].mean()) / df["score_testing"].std()

    grade_effect = df.loc[mask, "grade_applying_to"].apply(
        lambda g: 0.0 if g <= 5 else 0.3 if g <= 8 else 0.6
    )

    logit = (
        1.0
        + (0.8 + aid_slope) * df.loc[mask, "aid_offer_pct_tuition"]
        + 0.9 * df.loc[mask, "legacy_status"]
        + 0.6 * df.loc[mask, "tuition_enrolled_children"]
        + 0.25 * z_test_m
        + grade_effect
    )

    df.loc[mask, "enrolled"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    # 16. FINAL aid for new admits (need-based; honor the offer)
    AID_RATE = 0.18

    df["aid_offered_amount"] = 0.0
    df["aid_offered_pct_tuition"] = 0.0

    enrolled_mask = df["enrolled"] == 1
    n_enrolled = int(enrolled_mask.sum())
    target_n_aid = int(round(AID_RATE * n_enrolled))

    # Only enrolled students who were offered aid are eligible
    pool = df[enrolled_mask & (df["aid_offer_amount"] > 0)].copy()

    if len(pool) > 0 and target_n_aid > 0:
        target_n_aid = min(target_n_aid, len(pool))

        # Need-based weights
        w = (
            1.0
            + 3.0 * (pool["income_band"] == "<75k").astype(float)
            + 1.5 * (pool["income_band"] == "75–150k").astype(float)
            + 0.5 * (pool["income_band"] == "150–250k").astype(float)
            + 0.7 * (pool["family_size"] - 4).clip(lower=0)
            + 0.9 * pool["tuition_enrolled_children"].clip(lower=0)
            - 0.2 * pool["ses_centered"]
        ).clip(lower=0.01)

        probs = (w / w.sum()).to_numpy()
        chosen_idx = rng.choice(pool.index.to_numpy(), size=target_n_aid, replace=False, p=probs)

        # Honor the offer exactly
        df.loc[chosen_idx, "aid_offered_amount"] = df.loc[chosen_idx, "aid_offer_amount"]
        df.loc[chosen_idx, "aid_offered_pct_tuition"] = (
            df.loc[chosen_idx, "aid_offered_amount"] / df.loc[chosen_idx, "tuition"]
        ).clip(0, 1)


    return df

def simulate_many_years(
    n_years: int,
    applicants_per_year: int,
    zip_df: pd.DataFrame,
    base_seed: int = 42
) -> pd.DataFrame:
    all_years = []
    for year in range(1, n_years + 1):
        rng_year = np.random.default_rng(base_seed + year)
        df_year = simulate_applicants(applicants_per_year, zip_df, rng_year)
        df_year["year"] = year
        all_years.append(df_year)
    return pd.concat(all_years, ignore_index=True)
