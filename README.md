# AI-Based Placement Readiness Analytics

A Streamlit application for student placement readiness, machine-learning placement estimates, skill-gap suggestions, cohort analytics, and CSV exports for Power BI or Tableau.

## Files
- `app.py`: Streamlit interface, profile form, dynamic personalized dashboard, model metrics, and BI export page.
- `generate_data.py`: Generates the synthetic dataset.
- `placement_data.csv`: Dataset used by the app (include this file in the GitHub repository).
- `requirements.txt`: Python dependencies.

## Run on Windows
Open Command Prompt or the VS Code terminal in this folder:

```cmd
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If `placement_data.csv` is missing, generate it with:

```cmd
python generate_data.py
```

## How to make the dashboard change
1. Open **Student Assessment**.
2. Change the profile fields.
3. Click **Save Profile & Analyze**.
4. Open **Analytics Dashboard**. The personal metrics and skill comparison use the saved profile.
5. The historical cohort statistics use the dataset and the branch/tier filters, so those do not change just because a student profile changed.

## Power BI / Tableau
Open the **Power BI / Tableau Export** page and download the full dataset or the current student's profile. In Power BI Desktop use **Home → Get data → Text/CSV**. In Tableau use **Connect → Text File**. Download a new profile CSV after changing the profile to update your BI report. The app exports data for these tools; it does not embed a live Power BI or Tableau report.

## Important note
The placement dataset is synthetic demo data. Predictions are educational estimates and are not guarantees of real placement outcomes. Replace the synthetic dataset with a properly sourced, permissioned dataset before making real-world decisions.
