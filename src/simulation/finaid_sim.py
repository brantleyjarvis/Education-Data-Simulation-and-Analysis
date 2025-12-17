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
    TARGET_YIELD = 0.65          # expected yield among offers
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

    # 15. Enrollment decision
    df["enrolled"] = 0
    mask = df["spot_offered"] == 1

    df.loc[mask, "aid_offer_pct_tuition"] = df.loc[mask, "aid_offer_pct_tuition"].fillna(0.0)
    
    aid_slope = df.loc[mask, "income_band"].map({
        "<75k": 3.5, "75–150k": 2.8,
        "150–250k": 1.6, ">250k": 0.8
    }).fillna(0.0)  #  prevents NaN if income_band is unexpected

    z_test_m = (df.loc[mask, "score_testing"] - df["score_testing"].mean()) / df["score_testing"].std()

    grade_effect = df.loc[mask, "grade_applying_to"].apply(
        lambda g: 0.0 if g <= 5 else 0.3 if g <= 8 else 0.6
    )

    logit = (
        1.8
        + (0.8 + aid_slope) * df.loc[mask, "aid_offer_pct_tuition"]
        + 0.9 * df.loc[mask, "legacy_status"]
        + 0.6 * df.loc[mask, "tuition_enrolled_children"]
        + 0.25 * z_test_m
        + grade_effect
    )

    df.loc[mask, "enrolled"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    
    # 15.5 Waitlist fill to meet seat quotas (sweet-spot version)
    
    TARGET_FILL_RATE = 0.90       # minimum acceptable fill
    NEAR_MISS_DELTA = 0.6        # how far below cutoff we allow (tunable)
    ABSOLUTE_FLOOR = -1.25        # never admit below this admit_index
    
    for g, seats in seat_quota.items():
    
        sub = df[df["grade_applying_to"] == g].copy()
        if len(sub) == 0:
            continue
    
        # Current enrollment
        enrolled_idx = sub[sub["enrolled"] == 1].index
        current_fill = len(enrolled_idx) / seats
    
        if current_fill >= TARGET_FILL_RATE:
            continue
    
        shortfall = int(np.ceil(TARGET_FILL_RATE * seats - len(enrolled_idx)))
        if shortfall <= 0:
            continue
    
        # Recompute grade-specific cutoff (same as Step 13)
        cutoff = max(
            sub["admit_index"].quantile(CUTOFF_QUANTILE),
            MIN_CUTOFF
        )
    
        # Tier 1: standard admissible waitlist
        tier1 = sub[
            (sub["enrolled"] == 0) &
            (sub["admit_index"] >= cutoff)
        ].sort_values("admit_index", ascending=False)
    
        fill_idx = []
    
        if len(tier1) > 0:
            take = min(shortfall, len(tier1))
            fill_idx.extend(tier1.head(take).index.tolist())
            shortfall -= take
    
        # Tier 2: near-miss band (controlled relaxation)
        if shortfall > 0:
            tier2 = sub[
                (sub["enrolled"] == 0) &
                (sub["admit_index"] < cutoff) &
                (sub["admit_index"] >= max(cutoff - NEAR_MISS_DELTA, ABSOLUTE_FLOOR))
            ].sort_values("admit_index", ascending=False)
    
            if len(tier2) > 0:
                take = min(shortfall, len(tier2))
                fill_idx.extend(tier2.head(take).index.tolist())
    
        if len(fill_idx) == 0:
            continue
    
        # Force-enroll selected waitlist candidates
        df.loc[fill_idx, "spot_offered"] = 1
        df.loc[fill_idx, "enrolled"] = 1
    
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
