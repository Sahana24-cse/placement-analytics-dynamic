"""Create a synthetic placement dataset for the demo app."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate(n=100000, seed=42):
    rng = np.random.default_rng(seed)
    branches = ["CSE", "IT", "ECE", "ME", "EE", "CE", "Chemical"]
    branch_bonus = {"CSE": 0.35, "IT": 0.33, "ECE": 0.20, "ME": 0.05,
                    "EE": 0.05, "CE": -0.05, "Chemical": -0.07}
    df = pd.DataFrame({
        "branch": rng.choice(branches, n, p=[.2, .15, .15, .15, .12, .12, .11]),
        "college_tier": rng.choice(["Tier-1", "Tier-2", "Tier-3"], n, p=[.2, .4, .4]),
        "cgpa": np.clip(rng.normal(7.2, 1.0, n), 5, 10).round(2),
        "backlogs": rng.choice([0, 1, 2, 3, 4], n, p=[.6, .2, .1, .06, .04]),
        "coding_skills": np.clip(rng.normal(6, 1.8, n), 0, 10).round(1),
        "dsa_score": np.clip(rng.normal(6, 1.9, n), 0, 10).round(1),
        "aptitude_score": np.clip(rng.normal(65, 14, n), 0, 100).round(1),
        "communication_skills": np.clip(rng.normal(6.5, 1.7, n), 0, 10).round(1),
        "ml_knowledge": np.clip(rng.normal(5, 2, n), 0, 10).round(1),
        "system_design": np.clip(rng.normal(4.5, 2, n), 0, 10).round(1),
        "internships": rng.choice([0, 1, 2, 3], n, p=[.35, .35, .2, .1]),
        "projects_count": rng.choice([0, 1, 2, 3, 4, 5], n, p=[.1, .25, .3, .2, .1, .05]),
        "certifications": rng.choice([0, 1, 2, 3, 4], n, p=[.25, .3, .25, .12, .08]),
        "hackathons": rng.choice([0, 1, 2, 3], n, p=[.5, .25, .15, .1]),
        "open_source_contributions": rng.choice([0, 1, 2, 3], n, p=[.6, .2, .12, .08]),
        "extracurriculars": rng.choice([0, 1, 2, 3], n, p=[.3, .35, .25, .1]),
    })
    tier_bonus = df["college_tier"].map({"Tier-1": 0.4, "Tier-2": 0.0, "Tier-3": -0.1})
    z = (
        0.9 * (df["cgpa"] - 7.2) - 0.45 * df["backlogs"]
        + 0.18 * (df["coding_skills"] - 6) + 0.15 * (df["dsa_score"] - 6)
        + 0.02 * (df["aptitude_score"] - 65) + 0.10 * (df["communication_skills"] - 6.5)
        + 0.35 * df["internships"] + 0.12 * df["projects_count"]
        + 0.10 * df["certifications"] + 0.08 * df["hackathons"]
        + 0.05 * df["open_source_contributions"]
        + df["branch"].map(branch_bonus) + tier_bonus + rng.normal(0, 1.6, n)
    )
    # Choose an intercept that gives a target synthetic placement rate near 68.5%.
    lo, hi = float(z.min() - 10), float(z.max() + 10)
    for _ in range(60):
        threshold = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z - threshold)))).mean() > 0.685:
            lo = threshold
        else:
            hi = threshold
    probability = 1 / (1 + np.exp(-(z - threshold)))
    df["placed"] = (rng.random(n) < probability).astype(int)

    base_salary = df["college_tier"].map({"Tier-1": 19.3, "Tier-2": 17.1, "Tier-3": 16.5})
    salary = base_salary + 0.8 * (df["cgpa"] - 7.2) + 0.3 * (df["coding_skills"] - 6) + rng.normal(0, 2.2, n)
    df["salary_package_lpa"] = np.where(df["placed"] == 1, np.clip(salary, 6, 30).round(2), 0.0)
    return df


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "placement_data.csv"
    data = generate()
    data.to_csv(output, index=False)
    print(f"Saved {len(data):,} rows to {output}")
    print(f"Synthetic placement rate: {data['placed'].mean() * 100:.1f}%")
