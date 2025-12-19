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


# -----------------------------
# 2) Enforce offers + enrollment from targets
# -----------------------------
def apply_offers_and_enrollment_from_targets(
    df_year: pd.DataFrame,
    targets: pd.DataFrame,
    year: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Enforce grade-by-year offers/enrollment totals from targets.
    Who gets offered: top admit_index.
    Who enrolls: sampled among offered weighted by enrollment propensity.
    """

    # ---- COLUMN MAP (edit ONLY if your targets use different names) ----
    COL_YEAR = "year"
    COL_GRADE = "grade"
    COL_APPS = "apps"              # required (already used earlier)
    COL_OFFERS = "offers"          # preferred if present
    COL_ENROLLED = "enrolled"      # preferred if present
    COL_YIELD = "yield"            # optional fallback (0–1)
    COL_SEATS = "seats"            # optional fallback if offers not present

    t_year = targets[targets[COL_YEAR] == year].copy()

    if COL_OFFERS not in targets.columns and not (COL_SEATS in targets.columns and COL_YIELD in targets.columns):
        raise ValueError("Targets must have 'offers' OR (seats AND yield).")
    if COL_ENROLLED not in targets.columns and COL_YIELD not in targets.columns:
        raise ValueError("Targets must have 'enrolled' OR 'yield'.")

    df = df_year.copy()
    df["spot_offered"] = 0
    df["enrolled"] = 0

    # Admit index (same spirit as your original)
    z_test = (df["score_testing"] - df["score_testing"].mean()) / df["score_testing"].std()
    z_int  = (df["score_interview"] - df["score_interview"].mean()) / df["score_interview"].std()
    z_lead = (df["score_leadership"] - df["score_leadership"].mean()) / df["score_leadership"].std()

    income_bonus = df["income_band"].map({
        "<75k": 0.6,
        "75–150k": 0.3,
        "150–250k": 0.1,
        ">250k": 0.0
    }).fillna(0.0)

    df["admit_index"] = (
        0.6 * z_test +
        0.3 * z_int +
        0.3 * z_lead +
        0.6 * df["legacy_status"] +
        income_bonus
    )

    # Enrollment propensity (used only for choosing who enrolls, not to set totals)
    BASE_INTERCEPT = -0.6
    aid_slope = df["income_band"].map({
        "<75k": 3.5, "75–150k": 2.8,
        "150–250k": 1.6, ">250k": 0.8
    }).fillna(0.0)

    grade_effect = df["grade_applying_to"].apply(lambda g: 0.0 if g <= 5 else 0.3 if g <= 8 else 0.6)
    z_test_m = (df["score_testing"] - df["score_testing"].mean()) / df["score_testing"].std()

    # Offer pct is 0 here (aid comes later); still gives reasonable ranking by non-aid factors
    offer_pct = pd.Series(0.0, index=df.index)

    logit = (
        BASE_INTERCEPT
        + (0.8 + aid_slope) * offer_pct
        + 0.9 * df["legacy_status"]
        + 0.6 * df["tuition_enrolled_children"]
        + 0.25 * z_test_m
        + grade_effect
    )
    df["_enroll_prob"] = 1 / (1 + np.exp(-logit))

    # Apply grade-specific targets
    for g in range(1, 13):
        sub_idx = df.index[df["grade_applying_to"] == g]
        if len(sub_idx) == 0:
            continue

        row = t_year[t_year[COL_GRADE] == g]
        if row.empty:
            continue
        row = row.iloc[0]

        # Offers target
        if COL_OFFERS in targets.columns and pd.notna(row.get(COL_OFFERS, np.nan)):
            n_offers = int(row[COL_OFFERS])
        else:
            seats = int(row[COL_SEATS])
            y = float(row[COL_YIELD])
            y = max(min(y, 0.999), 0.01)
            n_offers = int(np.ceil(seats / y))

        n_offers = max(0, min(n_offers, len(sub_idx)))

        offered_idx = (
            df.loc[sub_idx]
            .sort_values("admit_index", ascending=False)
            .head(n_offers)
            .index
        )
        df.loc[offered_idx, "spot_offered"] = 1

        # Enrolled target
        if COL_ENROLLED in targets.columns and pd.notna(row.get(COL_ENROLLED, np.nan)):
            n_enrolled = int(row[COL_ENROLLED])
        else:
            y = float(row[COL_YIELD])
            y = max(min(y, 1.0), 0.0)
            n_enrolled = int(round(y * n_offers))

        n_enrolled = max(0, min(n_enrolled, len(offered_idx)))

        if n_enrolled > 0:
            probs = df.loc[offered_idx, "_enroll_prob"].to_numpy(dtype=float)
            probs = np.clip(probs, 1e-9, None)
            probs = probs / probs.sum()
            chosen = rng.choice(offered_idx.to_numpy(), size=n_enrolled, replace=False, p=probs)
            df.loc[chosen, "enrolled"] = 1

    df.drop(columns=["_enroll_prob"], inplace=True)
    return df


# -----------------------------
# 3) Multi-year runner from targets
# -----------------------------
def simulate_many_years_from_targets(
    targets: pd.DataFrame,
    zip_df: pd.DataFrame,
    base_seed: int = 42
) -> pd.DataFrame:
    all_years = []

    # basic schema checks (fail fast)
    required = {"year", "grade", "apps"}
    missing = required.difference(set(targets.columns))
    if missing:
        raise ValueError(f"Targets missing required columns: {missing}")

    for year in sorted(targets["year"].unique()):
        rng_year = np.random.default_rng(base_seed + int(year))

        df_year_parts = []
        for g in range(1, 13):
            row = targets[(targets["year"] == year) & (targets["grade"] == g)]
            if row.empty:
                continue

            n_apps = int(row["apps"].iloc[0])
            if n_apps <= 0:
                continue

            df_g = simulate_applicants(
                n_apps,
                zip_df,
                rng_year,
                forced_grade=g
            )
            df_g["year"] = int(year)
            df_year_parts.append(df_g)

        if not df_year_parts:
            continue

        df_year = pd.concat(df_year_parts, ignore_index=True)

        # enforce offers+enrollment totals for this year
        df_year = apply_offers_and_enrollment_from_targets(df_year, targets, int(year), rng_year)

        all_years.append(df_year)

    if not all_years:
        return pd.DataFrame()

    return pd.concat(all_years, ignore_index=True)


# -----------------------------
# 4) (Optional) Quick check helper
# -----------------------------
def check_against_targets(sim: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    chk = (
        sim.groupby(["year", "grade_applying_to"])
           .agg(apps=("applicant_id", "size"),
                offers=("spot_offered", "sum"),
                enrolled=("enrolled", "sum"))
           .reset_index()
           .rename(columns={"grade_applying_to": "grade"})
    )
    return chk.merge(targets, on=["year", "grade"], how="left")


    # 13. Spot offered (capacity-driven admissions)
    
    # Seat quotas by grade
    seat_quota = {
        1: 40, 2: 15, 3: 15, 4: 12, 5: 12,
        6: 30, 7: 8, 8: 8, 9: 20,
        10: 3, 11: 3, 12: 4
    }
    
    # Reset offers
    df["spot_offered"] = 0
    
    # Construct admit index
    z_test = (df["score_testing"] - df["score_testing"].mean()) / df["score_testing"].std()
    z_int = (df["score_interview"] - df["score_interview"].mean()) / df["score_interview"].std()
    z_lead = (df["score_leadership"] - df["score_leadership"].mean()) / df["score_leadership"].std()
    
    income_bonus = df["income_band"].map({
        "<75k": 0.6,
        "75–150k": 0.3,
        "150–250k": 0.1,
        ">250k": 0.0
    })
    
    df["admit_index"] = (
        0.6 * z_test +
        0.3 * z_int +
        0.3 * z_lead +
        0.6 * df["legacy_status"] +
        income_bonus
    )
    
    # Admissions tuning parameters
    TARGET_YIELD = 0.55          # expected yield among offers
    MAX_OFFER_MULTIPLIER = 1.8   # guardrail against absurd over-offering
    
    CUTOFF_QUANTILE = 0.55       # grade-specific quality cutoff
    MIN_CUTOFF = -0.75           # absolute floor on admit_index
    
    # Offer logic by grade
    for g, seats in seat_quota.items():
    
        sub = df[df["grade_applying_to"] == g].copy()
        if len(sub) == 0:
            continue
    
        # Grade-specific admissibility cutoff
        cutoff = max(
            sub["admit_index"].quantile(CUTOFF_QUANTILE),
            MIN_CUTOFF
        )
    
        admissible = sub[sub["admit_index"] >= cutoff].copy()
    
        # Fallback if cutoff is too strict for small grades
        if len(admissible) == 0:
            admissible = sub.sort_values("admit_index", ascending=False)
    
        # Back-solve number of offers
        n_to_offer = int(np.ceil(seats / TARGET_YIELD))
    
        n_to_offer = min(
            n_to_offer,
            int(MAX_OFFER_MULTIPLIER * seats),
            len(admissible)
        )
    
        # Select top admits
        offer_idx = admissible.sort_values(
            "admit_index", ascending=False
        ).head(n_to_offer).index
    
        df.loc[offer_idx, "spot_offered"] = 1

    # 14. Aid OFFER (used to drive enrollment decision)

    df["aid_offer_amount"] = 0.0
    df["aid_offer_pct_tuition"] = 0.0

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
            "<=50":    dict(p_award=0.95, mean=22474, low=10000, high=26500),
            "50-100":  dict(p_award=0.90, mean=17258, low=4750,  high=26750),
            "100-150": dict(p_award=0.85, mean=7327,  low=8000,  high=23250),
            ">150":    dict(p_award=0.00, mean=0,     low=0,     high=0),
        }
        
    MULTI = {
            "<=50":    dict(p_award=0.98, mean=24137, low=20000, high=31500),
            "50-100":  dict(p_award=0.95, mean=19902, low=10250, high=26750),
            "100-150": dict(p_award=0.90, mean=14494, low=1000,  high=28000),
            ">150":    dict(p_award=0.00, mean=0,     low=0,     high=0),
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

    # Generate aid OFFER as % of tuition (right-skewed)
    #   mean ≈ 0.65
    #   right-skewed toward higher aid
    A, B = 6.5, 3.0   # mean = A / (A + B) ≈ 0.68
    
    aid_pct = rng.beta(A, B, size=len(awarded))
    
    # Increase skew for multi-child households
    aid_pct *= np.where(awarded["multi_child"], 1.08, 1.0)
    
    # Clip to realistic bounds
    aid_pct = aid_pct.clip(0.20, 1.00)
    
    # Convert to dollars
    tuition = awarded["tuition"].to_numpy(dtype=float)
    offer_amt = aid_pct * tuition
    
    tuition = awarded["tuition"].to_numpy(dtype=float)
    offer_amt = np.minimum(offer_amt, tuition)

    df.loc[awarded.index, "aid_offer_amount"] = offer_amt
    df.loc[awarded.index, "aid_offer_pct_tuition"] = (offer_amt / tuition).clip(0, 1)

    # initialize final aid columns (will be overwritten post-enrollment)
    df["aid_offered_amount"] = 0.0
    df["aid_offered_pct_tuition"] = 0.0

    # 15. Enrollment decision (initial offers)
    df["enrolled"] = 0
    mask = df["spot_offered"] == 1

    BASE_INTERCEPT = -0.6  # keep consistent across initial + waitlist rounds

    df.loc[mask, "aid_offer_pct_tuition"] = df.loc[mask, "aid_offer_pct_tuition"].fillna(0.0).clip(0, 1)

    aid_slope = df.loc[mask, "income_band"].map({
        "<75k": 3.5, "75–150k": 2.8,
        "150–250k": 1.6, ">250k": 0.8
    }).fillna(0.0)

    z_test_m = (df.loc[mask, "score_testing"] - df["score_testing"].mean()) / df["score_testing"].std()

    grade_effect = df.loc[mask, "grade_applying_to"].apply(
        lambda g: 0.0 if g <= 5 else 0.3 if g <= 8 else 0.6
    )

    logit = (
        BASE_INTERCEPT
        + (0.8 + aid_slope) * df.loc[mask, "aid_offer_pct_tuition"]
        + 0.9 * df.loc[mask, "legacy_status"]
        + 0.6 * df.loc[mask, "tuition_enrolled_children"]
        + 0.25 * z_test_m
        + grade_effect
    )

    df.loc[mask, "enrolled"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    # 15.5–15.7 WAITLIST ROUNDS: offer -> (aid offer if needed) -> enroll
    MAX_WAITLIST_ROUNDS = 3
    TARGET_FILL_RATE = 0.90
    NEAR_MISS_DELTA = 0.6
    ABSOLUTE_FLOOR = -1.25

    for round_i in range(MAX_WAITLIST_ROUNDS):

        any_shortfall = False

        # --- offer additional students by grade ---
        for g, seats in seat_quota.items():

            sub = df[df["grade_applying_to"] == g].copy()
            if len(sub) == 0:
                continue

            enrolled_idx = sub[sub["enrolled"] == 1].index
            current_fill = len(enrolled_idx) / seats

            if current_fill >= TARGET_FILL_RATE:
                continue

            any_shortfall = True

            shortfall = int(np.ceil(TARGET_FILL_RATE * seats - len(enrolled_idx)))
            if shortfall <= 0:
                continue

            cutoff = max(
                sub["admit_index"].quantile(CUTOFF_QUANTILE),
                MIN_CUTOFF
            )

            tier1 = sub[
                (sub["spot_offered"] == 0) &
                (sub["admit_index"] >= cutoff)
            ].sort_values("admit_index", ascending=False)

            fill_idx = []

            if len(tier1) > 0:
                take = min(shortfall, len(tier1))
                fill_idx.extend(tier1.head(take).index.tolist())
                shortfall -= take

            if shortfall > 0:
                tier2 = sub[
                    (sub["spot_offered"] == 0) &
                    (sub["admit_index"] < cutoff) &
                    (sub["admit_index"] >= max(cutoff - NEAR_MISS_DELTA, ABSOLUTE_FLOOR))
                ].sort_values("admit_index", ascending=False)

                if len(tier2) > 0:
                    take = min(shortfall, len(tier2))
                    fill_idx.extend(tier2.head(take).index.tolist())

            if len(fill_idx) == 0:
                continue

            df.loc[fill_idx, "spot_offered"] = 1

        if not any_shortfall:
            break

        # --- generate aid offers for newly offered (still not enrolled) ---
        need_offer = (
            (df["spot_offered"] == 1) &
            (df["enrolled"] == 0) &
            (df["aid_requested"] == 1) &
            (df["aid_offer_amount"] == 0) &
            (df["income_band"].isin(["<75k", "75–150k"]))
        )

        idx = df.index[need_offer]
        if len(idx) > 0:
            A, B = 6.5, 3.0
            aid_pct = rng.beta(A, B, size=len(idx)).clip(0.20, 1.00)

            mc = (df.loc[idx, "tuition_enrolled_children"].to_numpy() >= 2)
            aid_pct = (aid_pct * np.where(mc, 1.08, 1.0)).clip(0.20, 1.00)

            tuition = df.loc[idx, "tuition"].to_numpy(dtype=float)
            offer_amt = np.minimum(aid_pct * tuition, tuition)

            df.loc[idx, "aid_offer_amount"] = offer_amt
            df.loc[idx, "aid_offer_pct_tuition"] = (offer_amt / tuition).clip(0, 1)

        # --- enroll newly offered (still enrolled==0) ---
        mask_new_offer = (df["spot_offered"] == 1) & (df["enrolled"] == 0)

        if mask_new_offer.any():
            df.loc[mask_new_offer, "aid_offer_pct_tuition"] = (
                df.loc[mask_new_offer, "aid_offer_pct_tuition"].fillna(0.0).clip(0, 1)
            )

            aid_slope_new = df.loc[mask_new_offer, "income_band"].map({
                "<75k": 3.5, "75–150k": 2.8,
                "150–250k": 1.6, ">250k": 0.8
            }).fillna(0.0)

            z_test_new = (df.loc[mask_new_offer, "score_testing"] - df["score_testing"].mean()) / df["score_testing"].std()

            grade_effect_new = df.loc[mask_new_offer, "grade_applying_to"].apply(
                lambda g: 0.0 if g <= 5 else 0.3 if g <= 8 else 0.6
            )

            logit_new = (
                BASE_INTERCEPT
                + (0.8 + aid_slope_new) * df.loc[mask_new_offer, "aid_offer_pct_tuition"]
                + 0.9 * df.loc[mask_new_offer, "legacy_status"]
                + 0.6 * df.loc[mask_new_offer, "tuition_enrolled_children"]
                + 0.25 * z_test_new
                + grade_effect_new
            )

            df.loc[mask_new_offer, "enrolled"] = rng.binomial(1, 1 / (1 + np.exp(-logit_new)))
    
    # HARD CAP: if a grade over-enrolls, keep only top admit_index enrollees up to seats
    for g, seats in seat_quota.items():
        enrolled_g = df[(df["grade_applying_to"] == g) & (df["enrolled"] == 1)]
        if len(enrolled_g) > seats:
            keep_idx = enrolled_g.sort_values("admit_index", ascending=False).head(seats).index
            drop_idx = enrolled_g.index.difference(keep_idx)
            df.loc[drop_idx, "enrolled"] = 0

    # 16. FINAL aid for new admits (need-based; honor the offer)
    #     POLICY TARGET:
    #       - ~85% of ENROLLED aid requesters (income < $150k) receive aid
    #       - Must have received a positive aid OFFER
    #       - If selected, realized aid == offer (exactly)
    #     Rename variables at end to avoid confusion
    
    TARGET_REQUESTER_AID_RATE = 0.85  # << key change: align to known data
    
    # 16a) Harden offer columns (critical: no NaNs) ----
    if "aid_offer_amount" not in df.columns:
        df["aid_offer_amount"] = 0.0
    if "aid_offer_pct_tuition" not in df.columns:
        df["aid_offer_pct_tuition"] = 0.0
    
    df["aid_offer_amount"] = pd.to_numeric(df["aid_offer_amount"], errors="coerce").fillna(0.0)
    df["aid_offer_pct_tuition"] = (
        pd.to_numeric(df["aid_offer_pct_tuition"], errors="coerce")
          .fillna(0.0)
          .clip(0, 1)
    )
    
    # 16b) Initialize REALIZED aid columns (post-enrollment dollars actually paid) 
    df["aid_offered_amount"] = 0.0
    df["aid_offered_pct_tuition"] = 0.0
    
    # Identify enrolled students
    enrolled_mask = df["enrolled"] == 1
    
    # Eligible pool (IMPORTANT CHANGE):
    # - enrolled
    # - requested aid
    # - positive offer
    # - income < $150k (and therefore 0 awards above 150k)
    eligible_pool = df[
        enrolled_mask
        & (df["aid_requested"] == 1)
        & (df["aid_offer_amount"] > 0)
        & (df["income_band"].isin(["<75k", "75–150k"]))
    ].copy()
    
    # Target number of aid recipients among eligible requesters
    target_n_aid = int(round(TARGET_REQUESTER_AID_RATE * len(eligible_pool)))
    
    # 16c) Need-based selection + honor offer exactly
    if len(eligible_pool) > 0 and target_n_aid > 0:
        target_n_aid = min(target_n_aid, len(eligible_pool))
    
        w = (
            1.0
            + 3.0 * (eligible_pool["income_band"] == "<75k").astype(float)
            + 1.5 * (eligible_pool["income_band"] == "75–150k").astype(float)
            + 0.7 * (eligible_pool["family_size"] - 4).clip(lower=0)
            + 0.9 * eligible_pool["tuition_enrolled_children"].clip(lower=0)
            - 0.2 * eligible_pool["ses_centered"]
        )
    
        # Make weights robust to NaNs and non-positive totals
        w = pd.to_numeric(w, errors="coerce").fillna(0.0).clip(lower=0.01)
        wsum = float(w.sum())
    
        if not np.isfinite(wsum) or wsum <= 0:
            probs = np.ones(len(w), dtype=float) / len(w)
        else:
            probs = (w / wsum).to_numpy(dtype=float)
    
        chosen_idx = rng.choice(
            eligible_pool.index.to_numpy(),
            size=target_n_aid,
            replace=False,
            p=probs
        )
    
        # REALIZED aid = OFFER (exactly), only for chosen (enrolled) recipients
        df.loc[chosen_idx, "aid_offered_amount"] = df.loc[chosen_idx, "aid_offer_amount"]
    
        tuition_chosen = pd.to_numeric(df.loc[chosen_idx, "tuition"], errors="coerce").fillna(0.0)
    
        df.loc[chosen_idx, "aid_offered_amount"] = np.minimum(
            df.loc[chosen_idx, "aid_offered_amount"].to_numpy(dtype=float),
            tuition_chosen.to_numpy(dtype=float)
        )
    
        df.loc[chosen_idx, "aid_offered_pct_tuition"] = np.where(
            tuition_chosen.to_numpy(dtype=float) > 0,
            (df.loc[chosen_idx, "aid_offered_amount"].to_numpy(dtype=float) / tuition_chosen.to_numpy(dtype=float)),
            0.0
        )
    
        df.loc[chosen_idx, "aid_offered_pct_tuition"] = (
            pd.Series(df.loc[chosen_idx, "aid_offered_pct_tuition"])
              .fillna(0.0)
              .clip(0, 1)
              .to_numpy()
        )
    
    # 16d) Rename variables to avoid confusion
    # offer_*     = what was offered pre-enrollment (drives yield)
    # realized_*  = what is actually paid post-enrollment (use for budget/EDA)
    df.rename(
        columns={
            "aid_offer_amount": "offer_amount",
            "aid_offer_pct_tuition": "offer_pct_tuition",
            "aid_offered_amount": "realized_aid_amount",
            "aid_offered_pct_tuition": "realized_aid_pct_tuition",
        },
        inplace=True
    )
    
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
            df_g = simulate_applicants(n_apps, zip_df, rng_year)
            df_g["year"] = year
            df_g["grade_applying_to"] = g  # override grade draw
            df_g["tuition"] = df_g["grade_applying_to"].map(tuition_by_grade)  # keep consistent
            df_year_parts.append(df_g)

        df_year = pd.concat(df_year_parts, ignore_index=True)

        # Apply observed offers/enrollment calibration
        df_year = apply_offers_and_enrollment_from_targets(df_year, targets, rng_year)

        all_years.append(df_year)

    return pd.concat(all_years, ignore_index=True)

