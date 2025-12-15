# Simulation Design Summary  
**Facts (F) and Assumptions (A)**

This simulation generates a synthetic applicant-to-enrollment pipeline for a private K–12 school. Where possible, empirical inputs are used (**Facts**). Remaining structure and relationships are informed by domain knowledge and stated explicitly as **Assumptions** (A).

---

## 2. Grade Applying To
- Overall application volume by grade is based on estimates provided by the school point of contact (**F**).
- Application volume peaks at **1st, 6th, and 9th grades**, consistent with common entry points (**F**).
- Outside of entry points, application volume declines with increasing grade level (**A**).

---

## 3. Tuition by Grade
- Tuition is mapped directly by grade using SY 2025–2026 tuition data (**F**).
- Tuition levels are assumed stable across time, with no major grade-specific shifts (**A**).

---

## 4. ZIP Assignment and Socioeconomic Status (SES)
- No direct data are available on applicant-level SES (**F**).
- Distribution of current enrolled students by city is available (**F**).
- The applicant pool is assumed to resemble the enrolled population geographically and to reflect ZIP-level SES distributions derived from ACS data (**A**).

---

## 5. Race / Ethnicity
- Race/ethnicity distributions are available by school level for enrolled students (**F**).
- The applicant pool is assumed to mirror enrolled distributions by level (**A**).
- A small SES-linked relationship (via ACS) is incorporated to reflect observed population patterns (**F**).

---

## 6. Gender
- Applicant gender is assumed to be evenly distributed (**A**).

---

## 7. Income Band
- Applicant household income is assumed to be **right-skewed**, with a greater share of high-income families relative to the general population (**A**).

---

## 8. Family Size
- Family size is assumed to be **right-skewed**, with applicants more likely to come from multi-child households (**A**).

---

## 9. Tuition-Enrolled Children
- The probability that a family already has a child enrolled and paying tuition increases with family size (**A**).
- A small positive SES effect is included to reflect greater ability to sustain multiple tuitions (**A**).

---

## 10. Legacy Status
- Legacy probability is influenced by:
  - Applicant city (proxy for alumni concentration) (**A**)
  - Grade applying to (legacy more common at early entry grades, especially 1st) (**A**)
  - SES (higher SES more likely to be legacy) (**A**)
  - Presence of another tuition-enrolled child (positive correlation) (**A**)

---

## 11. Aid Requested
- The decision to request aid is driven primarily by household income band (**F**).
- Additional modifiers include:
  - Family size (**A**)
  - Presence of another tuition-enrolled child (**A**)
  - SES (higher SES less likely to request aid) (**A**)

---

## 12. Applicant Scores
- Academic, interview, leadership, athletics, and arts scores are modeled as **moderately correlated latent traits** (**A**).
- Applicants from higher-SES areas score higher on average (**F**).
- Applicants with minority status score lower on average, consistent with observed population-level disparities (**F**).

---

## 13. Spot Offered (Admissions)
- Admission decisions are made via a grade-specific quota system.
- An explicit **income-based bonus** is applied in the admissions index to favor offering spots to lower-income applicants (**A**).

---

## 14. Aid Offered
- Financial aid awards are driven primarily by income band (**A**).
- Award amounts are adjusted for:
  - Family size (**A**)
  - SES (**A**)
  - Minority status (**A**)
  - Presence of multiple tuition-enrolled children (**A**)

---

## 15. Enrollment Decision (Yield)
- Enrollment probability is modeled using a logistic yield function.
- Low-income families are assumed to be more price-sensitive to aid generosity (**A**).
- Grades with fewer available seats are assumed to have higher yield rates (**A**).
- Legacy families and families with children already enrolled are more likely to accept an offer (**A**).
- Applicants with higher scores are more likely to enroll, reflecting overall “fit” (**A**).

---

*This framework is intended to produce realistic aggregate behavior rather than precise individual-level prediction. All assumptions are explicitly documented to support sensitivity testing and future refinement.*
