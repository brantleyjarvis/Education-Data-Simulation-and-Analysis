# Private Historical Targets (Not Committed)

This directory contains admissions data used in the simulation.

Expected file:
- `grade_year_targets.csv`

Required columns:
- year
- grade
- apps
- offers
- enr

Derived fields (computed in code):
- offer_rate = offers / apps
- yield_rate = enr / offers

To reproduce results without access to real data:
- substitute a synthetic or anonymized version
- preserve the same schema
