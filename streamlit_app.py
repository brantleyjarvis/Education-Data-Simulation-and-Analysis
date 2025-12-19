# src/dashboard/streamlit_app.py
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dashboard_backend import (
    COHORTS, get_cohort, build_pick_list,
    build_applicant_curve, min_aid_for_thresholds, p_at_award,
    optimize_budget
)

st.set_page_config(page_title="Aid Offer Explorer + Budget Optimizer", layout="wide")

@st.cache_data
def load_data():
    offers = pd.read_parquet("data/private/offers.parquet")
    marg_sorted = pd.read_parquet("data/private/marg_sorted.parquet")
    return offers, marg_sorted

@st.cache_resource
def load_model():
    return joblib.load("data/private/mono_model.joblib")

offers, marg_sorted = load_data()
mono_model = load_model()

st.title("Financial Aid Dashboard (Demo)")

tab1, tab2 = st.tabs(["Applicant Offer Explorer", "Budget Optimizer"])

# --------------------
# Tab 1: Applicant Explorer
# --------------------
with tab1:
    st.subheader("Pick an applicant")
    c1, c2 = st.columns(2)
    with c1:
        cohort_name = st.selectbox("Cohort", list(COHORTS.keys()), index=0)
    with c2:
        mode = st.radio("Select by", ["Applicant ID", "Filters"], horizontal=True)

    if mode == "Applicant ID":
        applicant_id = st.number_input("Applicant ID", min_value=1, value=1, step=1)
        # map applicant_id -> row_id in current cohort
        offers_sub, marg_sub = get_cohort(offers, marg_sorted, cohort_name)
        match = offers_sub[offers_sub["applicant_id"] == int(applicant_id)]
        if match.empty:
            st.warning("Applicant not found in this cohort.")
            st.stop()
        row_id = int(match["row_id"].iloc[0])
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            grade = st.selectbox("Grade applying to", [None] + sorted(offers["grade_applying_to"].unique().tolist()))
            income_band = st.selectbox("Income band", [None] + sorted(offers["income_band"].dropna().unique().tolist()))
        with f2:
            minority = st.selectbox("Minority flag", [None, 0, 1])
            legacy = st.selectbox("Legacy", [None, 0, 1])
        with f3:
            aid_requested = st.selectbox("Aid requested", [None, 0, 1])
            n_pick = st.slider("Pick list size", 5, 25, 15)

        pick = build_pick_list(
            offers, cohort_name=cohort_name, grade=grade, income_band=income_band,
            minority=minority, legacy=legacy, aid_requested=aid_requested, n=n_pick
        )
        st.dataframe(pick, use_container_width=True, height=260)

        if pick.empty:
            st.stop()

        row_id = int(pick.loc[0, "row_id"])
        st.caption(f"Using first row in pick list (row_id={row_id}). In production, click-to-select.")

    # Build curve + anchors
    curve = build_applicant_curve(offers, marg_sorted, mono_model, row_id=row_id, cohort_name=cohort_name)
    anchors = min_aid_for_thresholds(curve, step=0.10)

    # Profile card
    offers_sub, _ = get_cohort(offers, marg_sorted, cohort_name)
    prof = offers_sub[offers_sub["row_id"] == row_id].iloc[0].to_dict()

    st.markdown("### Applicant profile")
    st.write({
        "applicant_id": int(prof.get("applicant_id")),
        "grade_applying_to": int(prof.get("grade_applying_to")),
        "tuition": float(prof.get("tuition")),
        "income_band": prof.get("income_band"),
        "race_ethnic_minority": int(prof.get("race_ethnic_minority")),
        "legacy_status": int(prof.get("legacy_status")),
        "aid_requested": int(prof.get("aid_requested")),
        "admit_index": float(prof.get("admit_index")) if "admit_index" in prof else None,
    })

    # Slider: award dollars
    max_award = float(curve["aid_dollars"].max())
    award = st.slider("Aid offer ($)", 0.0, max_award if max_award > 0 else 50000.0, 0.0, step=500.0)

    snap = p_at_award(curve, award)
    st.metric("Predicted enrollment probability", f"{snap['p_enroll']:.3f}")
    st.caption(f"Snapped to nearest grid point: ${snap['aid_dollars_nearest']:.0f} (aid_pct={snap['aid_pct_nearest']:.2f})")

    # Plot curve
    st.markdown("### Aid → Enrollment probability curve")
    fig = plt.figure()
    plt.plot(curve["aid_dollars"], curve["p_enroll"], marker="o")
    plt.xlabel("Aid offer ($)")
    plt.ylabel("P(enroll)")
    plt.ylim(0, 1)
    st.pyplot(fig)

    st.markdown("### Minimum aid to reach thresholds (baseline → 100% by 10%)")
    st.dataframe(anchors, use_container_width=True)

# --------------------
# Tab 2: Budget Optimizer
# --------------------
with tab2:
    st.subheader("Optimize distribution given a budget")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cohort_name2 = st.selectbox("Cohort (optimizer)", list(COHORTS.keys()), index=0)
    with c2:
        budget = st.number_input("Budget ($)", min_value=0, value=3_000_000, step=100_000)
    with c3:
        min_low = st.slider("Min spend share: <75k", 0.0, 0.8, 0.35, step=0.05)
    with c4:
        min_div = st.slider("Min spend share: minority", 0.0, 0.8, 0.20, step=0.05)

    run = st.button("Run optimizer")

    if run:
        out = optimize_budget(
            offers_df=offers,
            marg_df=marg_sorted,
            model=mono_model,
            cohort_name=cohort_name2,
            budget=float(budget),
            min_share_low=float(min_low),
            min_share_div=float(min_div),
            diversity_col="race_ethnic_minority",
            low_income_band="<75k",
            top_n=25
        )

        st.markdown("### KPI")
        st.dataframe(out["kpi"], use_container_width=True)

        a, b = st.columns(2)
        with a:
            st.markdown("### Spend / impact by income band")
            st.dataframe(out["by_income"], use_container_width=True, height=300)
        with b:
            st.markdown("### Spend / impact by grade")
            st.dataframe(out["by_grade"], use_container_width=True, height=300)

        st.markdown("### Top recommendations (largest p gain)")
        cols_show = ["applicant_id","grade_applying_to","income_band","race_ethnic_minority","aid_dollar_final","aid_pct_implied","p_enroll_baseline","p_enroll_final"]
        st.dataframe(out["top_recs"][cols_show], use_container_width=True, height=350)
