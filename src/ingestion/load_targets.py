import pandas as pd


def load_grade_year_targets(path: str) -> pd.DataFrame:
    """
    Load historical applicant / offer / enrollment targets by grade-year.
    """
    targets = pd.read_csv(path)

    required_cols = {"year", "grade", "apps", "offers", "enr"}
    missing = required_cols - set(targets.columns)
    if missing:
        raise ValueError(f"Targets file missing columns: {missing}")

    # Type enforcement
    targets["year"] = targets["year"].astype(int)
    targets["grade"] = targets["grade"].astype(int)
    targets["apps"] = targets["apps"].astype(int)
    targets["offers"] = targets["offers"].astype(int)
    targets["enr"] = targets["enr"].astype(int)

    # Basic integrity checks
    if (targets["offers"] > targets["apps"]).any():
        raise ValueError("Found offers > apps in targets")

    if (targets["enr"] > targets["offers"]).any():
        raise ValueError("Found enrolled > offers in targets")

    # Derived rates (used later)
    targets["offer_rate"] = targets["offers"] / targets["apps"]
    targets["yield_rate"] = targets["enr"] / targets["offers"]

    return targets
