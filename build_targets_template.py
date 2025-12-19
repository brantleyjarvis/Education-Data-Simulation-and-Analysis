import pandas as pd

years = list(range(2018, 2026))   # 2018–2025
grades = list(range(1, 13))       # 1–12

rows = [{"year": y, "grade": g, "apps": "", "offers": "", "enr": ""} for y in years for g in grades]
df = pd.DataFrame(rows)

out_path = "data/private/grade_year_targets.csv"
print(df.head())
print(f"\nTemplate created in-memory. Save it to: {out_path}")
