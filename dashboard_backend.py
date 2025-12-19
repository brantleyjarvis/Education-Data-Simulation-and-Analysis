# src/dashboard/dashboard_backend.py
from __future__ import annotations

import numpy as np
import pandas as pd

COHORTS = {
    "All": None,
    "Academy entrants (Grade 1)": [1],
    "School entrants (Grades 1/6/9)": [1, 6, 9],
}

def get_cohort(offers_df: pd.DataFrame, marg_df: pd.DataFrame, cohort_name: str):
    grades = COHORTS[cohort_name]
    if grades is None:
        return offers_df.copy(), marg_df.copy()
    offers_sub = offers_df[offers_df["grade_applying_to"].isin(grades)].copy()
    keep = set(offers_sub["row_id"].tolist())
    marg_sub = marg_df[marg_df["row_id"].isin(keep)].copy()
    return offers_sub, marg_sub

def build_pick_list(
    offers_df: pd.DataFrame,
    cohort_name: str = "All",
    grade: int | None = None,
    income_band: str | None = None,
    minority: int | None = None,
    legacy: int | None = None,
    aid_requested: int | None = None,
    admit_index_min: float | None = None,
    admit_index_max: float | None = None,
    n: int = 15,
    seed: int = 7
) -> pd.DataFrame:
    df, _ = get_cohort(offers_df, pd.DataFrame({"row_id": offers_df["row_id"]}), cohort_name)

    if grade is not None:
        df = df[df["grade_applying_to"] == grade]
    if income_band is not None:
        df = df[df["income_band"] == income_band]
    if minority is not None:
        df = df[df["race_ethnic_minority"] == minority]
    if legacy is not None:
        df = df[df["legacy_status"] == legacy]
    if aid_requested is not None:
        df = df[df["aid_requested"] == aid_requested]

    if "admit_index" in df.columns:
        if admit_index_min is not None:
            df = df[df["admit_index"] >= admit_index_min]
        if admit_index_max is not None:
            df = df[df["admit_index"] <= admit_index_max]

    cols = [
        "applicant_id","row_id","grade_applying_to","income_band","race_ethnic_minority",
        "legacy_status","aid_requested","tuition"
    ]
    if "admit_index" in df.columns:
        cols.append("admit_index")

    if df.empty:
        return pd.DataFrame(columns=cols)

    rng = np.random.default_rng(seed)
    idx = df.index.to_numpy()
    take = min(n, len(idx))
    pick_idx = rng.choice(idx, size=take, replace=False)
    out = df.loc[pick_idx, cols].sort_values(
        ["grade_applying_to","income_band","race_ethnic_minority"],
        ascending=[True, True, False]
    )
    return out.reset_index(drop=True)

# ----------------------------
# Applicant explorer
# ----------------------------

def _build_X_aligned_for_model(df: pd.DataFrame, model, aid_pct_value: float) -> pd.DataFrame:
    feat = list(model.feature_names_in_)
    X = pd.DataFrame(0.0, index=df.index, columns=feat)

    overlap = [c for c in feat if c in df.columns]
    if overlap:
        X.loc[:, overlap] = df[overlap].astype(float)

    def norm_band(x):
        if pd.isna(x):
            return None
        return str(x).replace("–", "-").replace("—", "-").strip()

    band_map = {
        "<75k": "income_band_75k",
        "75–150k": "income_band_75-150k",
        "75-150k": "income_band_75-150k",
        "150–250k": "income_band_150-250k",
        "150-250k": "income_band_150-250k",
        ">250k": "income_band_250k",
    }

    if "income_band" in df.columns:
        bands = df["income_band"].map(norm_band)
        for raw, dummy in band_map.items():
            if dummy in X.columns:
                X.loc[bands == norm_band(raw), dummy] = 1.0

    if "aid_pct" in X.columns:
        X["aid_pct"] = float(aid_pct_value)
    elif "aid_pct_final" in X.columns:
        X["aid_pct_final"] = float(aid_pct_value)

    return X

def predict_p_at_aid_pct(offers_sub: pd.DataFrame, row_id: int, model, aid_pct: float) -> float:
    row = offers_sub.loc[offers_sub["row_id"] == row_id].copy()
    if row.empty:
        raise ValueError(f"row_id {row_id} not found")
    X = _build_X_aligned_for_model(row, model, aid_pct_value=aid_pct)
    return float(model.predict_proba(X)[:, 1][0])

def build_applicant_curve(offers_df, marg_df, model, row_id: int, cohort_name: str):
    offers_sub, marg_sub = get_cohort(offers_df, marg_df, cohort_name)

    df = marg_sub[marg_sub["row_id"] == row_id].sort_values("aid_pct").copy()

    # baseline at aid=0 from model
    p0 = predict_p_at_aid_pct(offers_sub, row_id, model, aid_pct=0.0)
    base = pd.DataFrame([{"row_id": row_id, "aid_pct": 0.0, "aid_dollars": 0.0, "p_enroll": p0}])

    if df.empty:
        return base

    df = df[["row_id","aid_pct","aid_dollars","p_enroll"]].copy()
    curve = pd.concat([base, df], ignore_index=True).drop_duplicates(["aid_pct"], keep="first")
    return curve.sort_values("aid_pct").reset_index(drop=True)

def thresholds_from_baseline(p0: float, step: float = 0.10) -> list[float]:
    p0 = float(np.clip(p0, 0, 1))
    start = np.ceil(p0 / step) * step
    if start <= p0 + 1e-12:
        start = min(1.0, start + step)

    thr = []
    x = start
    while x <= 1.0 + 1e-12:
        thr.append(round(min(1.0, x), 2))
        x += step
    if not thr and p0 < 1.0:
        thr = [1.0]
    return thr

def min_aid_for_thresholds(curve_df: pd.DataFrame, step: float = 0.10) -> pd.DataFrame:
    p0 = float(curve_df.loc[curve_df["aid_dollars"] == 0.0, "p_enroll"].iloc[0])
    thr = thresholds_from_baseline(p0, step=step)

    out = []
    for t in thr:
        hit = curve_df[curve_df["p_enroll"] >= t].sort_values("aid_dollars").head(1)
        if hit.empty:
            out.append({"threshold": t, "min_aid_dollars": np.nan, "min_aid_pct": np.nan, "p_enroll_at_min": np.nan})
        else:
            out.append({
                "threshold": t,
                "min_aid_dollars": float(hit["aid_dollars"].iloc[0]),
                "min_aid_pct": float(hit["aid_pct"].iloc[0]),
                "p_enroll_at_min": float(hit["p_enroll"].iloc[0]),
            })
    return pd.DataFrame(out)

def p_at_award(curve_df: pd.DataFrame, award_dollars: float) -> dict:
    df = curve_df.copy()
    idx = (df["aid_dollars"] - float(award_dollars)).abs().idxmin()
    r = df.loc[idx]
    return {"aid_dollars_nearest": float(r["aid_dollars"]), "aid_pct_nearest": float(r["aid_pct"]), "p_enroll": float(r["p_enroll"])}

# ----------------------------
# Portfolio optimizer (your constrained greedy logic)
# ----------------------------

def run_alloc_equity_div(
    marg_sorted_in: pd.DataFrame,
    offers_in: pd.DataFrame,
    budget: float,
    low_income_band: str = "<75k",
    min_share_low: float = 0.35,
    diversity_col: str = "race_ethnic_minority",
    min_share_div: float = 0.20,
) -> pd.DataFrame:
    ms = marg_sorted_in.copy()

    attrs_needed = ["income_band", "aid_requested", diversity_col]
    missing = [c for c in attrs_needed if c not in ms.columns]
    if missing:
        ms = ms.merge(offers_in[["row_id"] + missing], on="row_id", how="left")

    ms = ms[
        (ms["income_band"] != ">250k") &
        ~((ms["income_band"] == "150–250k") & (ms["aid_requested"] != 1))
    ].copy()

    ms = ms.sort_values("marginal_yield_per_dollar", ascending=False).copy()

    def allocate_steps(pool_df, budget_left, used_step_keys):
        pool_df = pool_df[~pool_df["step_key"].isin(used_step_keys)].copy()
        if len(pool_df) == 0 or budget_left <= 0:
            return pool_df.iloc[0:0].copy()
        pool_df["cum_spend"] = pool_df["delta_$"].cumsum()
        return pool_df[pool_df["cum_spend"] <= budget_left].copy()

    used = set()
    parts = []

    B_low = budget * min_share_low
    B_div = budget * min_share_div

    alloc_low = allocate_steps(ms[ms["income_band"] == low_income_band], B_low, used)
    used.update(alloc_low["step_key"].tolist())
    parts.append(alloc_low)
    spent_low = float(alloc_low["delta_$"].sum())

    alloc_div = allocate_steps(ms[ms[diversity_col] == 1], B_div, used)
    used.update(alloc_div["step_key"].tolist())
    parts.append(alloc_div)
    spent_div = float(alloc_div["delta_$"].sum())

    B_remain = max(0.0, budget - (spent_low + spent_div))
    alloc_rest = allocate_steps(ms, B_remain, used)
    used.update(alloc_rest["step_key"].tolist())
    parts.append(alloc_rest)

    return pd.concat(parts, ignore_index=True)

def build_offers_final_from_alloc(offers_in: pd.DataFrame, alloc_steps: pd.DataFrame) -> pd.DataFrame:
    final_pct = alloc_steps.groupby("row_id", as_index=False).agg(aid_pct_final=("aid_pct", "max"))
    final_dol = alloc_steps.groupby("row_id", as_index=False).agg(aid_dollar_final=("delta_$", "sum"))

    out = offers_in.copy().merge(final_pct, on="row_id", how="left").merge(final_dol, on="row_id", how="left")
    out["aid_pct_final"] = out["aid_pct_final"].fillna(0.0)
    out["aid_dollar_final"] = out["aid_dollar_final"].fillna(0.0)

    out.loc[out["income_band"] == ">250k", ["aid_pct_final","aid_dollar_final"]] = 0.0
    out.loc[(out["income_band"] == "150–250k") & (out["aid_requested"] != 1), ["aid_pct_final","aid_dollar_final"]] = 0.0

    out["aid_pct_implied"] = (out["aid_dollar_final"] / out["tuition"]).clip(0, 1)
    return out

def fill_probs_baseline_and_final(offers_final: pd.DataFrame, model) -> pd.DataFrame:
    """
    Simple + consistent for dashboard:
      baseline = model at aid=0
      final = model at aid=aid_pct_implied (or aid_pct_final if preferred)
    (This avoids rerun headaches with p_enroll_at_step.)
    """
    out = offers_final.copy()

    # baseline at 0
    X0 = _build_X_aligned_for_model(out, model, aid_pct_value=0.0)
    out["p_enroll_baseline"] = model.predict_proba(X0)[:, 1]

    # final at implied pct (dollars/tuition) – consistent with incremental spend outputs
    aid_pct_use = out["aid_pct_implied"].astype(float).clip(0, 1).to_numpy()
    p_final = np.zeros(len(out), dtype=float)
    # vectorize in chunks (build_X expects scalar, so do loop; fast enough for demo sizes)
    for i, ap in enumerate(aid_pct_use):
        Xi = _build_X_aligned_for_model(out.iloc[[i]], model, aid_pct_value=float(ap))
        p_final[i] = float(model.predict_proba(Xi)[:, 1][0])
    out["p_enroll_final"] = p_final

    return out

def optimize_budget(
    offers_df: pd.DataFrame,
    marg_df: pd.DataFrame,
    model,
    cohort_name: str,
    budget: float,
    min_share_low: float,
    min_share_div: float,
    diversity_col: str = "race_ethnic_minority",
    low_income_band: str = "<75k",
    top_n: int = 25
):
    offers_sub, marg_sub = get_cohort(offers_df, marg_df, cohort_name)

    alloc_steps = run_alloc_equity_div(
        marg_sorted_in=marg_sub,
        offers_in=offers_sub,
        budget=budget,
        low_income_band=low_income_band,
        min_share_low=min_share_low,
        diversity_col=diversity_col,
        min_share_div=min_share_div
    )

    offers_final = build_offers_final_from_alloc(offers_sub, alloc_steps)
    offers_final = fill_probs_baseline_and_final(offers_final, model)

    spend = float(offers_final["aid_dollar_final"].sum())
    delta = float((offers_final["p_enroll_final"] - offers_final["p_enroll_baseline"]).sum())

    kpi = pd.DataFrame([{
        "cohort": cohort_name,
        "n_offers": int(len(offers_final)),
        "budget": float(budget),
        "spend_final": spend,
        "exp_enroll_baseline": float(offers_final["p_enroll_baseline"].sum()),
        "exp_enroll_final": float(offers_final["p_enroll_final"].sum()),
        "exp_enroll_delta": delta,
        "cost_per_exp_enroll": spend / max(1e-9, delta),
        "spend_share_<75k": float(offers_final.loc[offers_final["income_band"] == "<75k","aid_dollar_final"].sum()) / max(1e-9, spend),
        "spend_share_minority": float(offers_final.loc[offers_final[diversity_col] == 1,"aid_dollar_final"].sum()) / max(1e-9, spend),
    }])

    by_income = (
        offers_final.groupby("income_band", as_index=False)
        .agg(spend=("aid_dollar_final","sum"),
             exp_enroll_delta=("p_enroll_final", "sum"))
    )
    by_income["spend_share"] = by_income["spend"] / max(1e-9, by_income["spend"].sum())

    by_grade = (
        offers_final.groupby("grade_applying_to", as_index=False)
        .agg(spend=("aid_dollar_final","sum"),
             exp_enroll_final=("p_enroll_final","sum"),
             exp_enroll_baseline=("p_enroll_baseline","sum"))
    )
    by_grade["exp_enroll_delta"] = by_grade["exp_enroll_final"] - by_grade["exp_enroll_baseline"]

    top_recs = offers_final.copy()
    top_recs["p_gain"] = top_recs["p_enroll_final"] - top_recs["p_enroll_baseline"]
    top_recs = top_recs.sort_values("p_gain", ascending=False).head(top_n)

    return {
        "kpi": kpi,
        "offers_final": offers_final,
        "alloc_steps": alloc_steps,
        "by_income": by_income.sort_values("spend_share", ascending=False),
        "by_grade": by_grade.sort_values("grade_applying_to"),
        "top_recs": top_recs,
    }
