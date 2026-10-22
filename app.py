"""AI-Based Placement Readiness Analytics
Run locally: python -m streamlit run app.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from generate_data import generate

st.set_page_config(
    page_title="Placement Readiness Analytics",
    page_icon="🎓",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "placement_data.csv"

CAT_FEATURES = ["branch", "college_tier"]
FEATURES = [
    "branch", "college_tier", "cgpa", "backlogs", "coding_skills", "dsa_score",
    "aptitude_score", "communication_skills", "ml_knowledge", "system_design",
    "internships", "projects_count", "certifications", "hackathons",
    "open_source_contributions", "extracurriculars",
]
NUM_FEATURES = [c for c in FEATURES if c not in CAT_FEATURES]
TARGET = "placed"


@st.cache_data
def load_data(file_stamp):
    """Load the local dataset; create it if it does not exist."""
    if not DATA_FILE.exists():
        data = generate()
        data.to_csv(DATA_FILE, index=False)
    data = pd.read_csv(DATA_FILE)
    required = set(FEATURES + [TARGET, "salary_package_lpa"])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    return data


@st.cache_resource
def train_models(data_signature):
    """Train three models once per dataset signature."""
    df = load_data(data_signature)
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUM_FEATURES),
        ],
        remainder="drop",
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
        ),
    }

    fitted, metric_rows = {}, []
    for name, classifier in classifiers.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", classifier)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metric_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1 score": f1_score(y_test, predictions, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, probabilities),
        })
        fitted[name] = pipeline

    metrics = pd.DataFrame(metric_rows).set_index("Model")
    best_model_name = metrics["ROC AUC"].idxmax()
    return fitted, metrics, best_model_name


def calculate_readiness(profile):
    academic = float(np.clip(profile["cgpa"] / 10 * 100, 0, 100))
    technical = float(np.mean([
        profile["coding_skills"], profile["dsa_score"],
        profile["ml_knowledge"], profile["system_design"],
    ]) * 10)
    aptitude = float(profile["aptitude_score"])
    communication = float(profile["communication_skills"] * 10)
    experience = min(
        profile["internships"] * 40 + profile["projects_count"] * 15
        + profile["certifications"] * 10 + profile["hackathons"] * 10
        + profile["open_source_contributions"] * 10,
        100,
    )
    components = {
        "Academic": academic,
        "Technical": technical,
        "Aptitude": aptitude,
        "Communication": communication,
        "Experience": float(experience),
    }
    total = (
        0.30 * academic + 0.30 * technical + 0.20 * aptitude
        + 0.10 * communication + 0.10 * experience
    )
    return round(total), {name: round(value) for name, value in components.items()}


def profile_to_frame(profile):
    return pd.DataFrame([{feature: profile[feature] for feature in FEATURES}])


def get_recommendations(profile):
    gaps, recommendations = [], []
    if profile["cgpa"] < 7.0:
        gaps.append("CGPA is below 7.0.")
        recommendations.append("Revise core subjects and aim to improve semester grades.")
    if profile["backlogs"] > 0:
        gaps.append(f"You have {profile['backlogs']} backlog(s).")
        recommendations.append("Prioritize clearing backlogs before placement drives.")
    if profile["coding_skills"] < 7:
        gaps.append("Coding skills need improvement.")
        recommendations.append("Practice one coding problem daily and revise Python fundamentals.")
    if profile["dsa_score"] < 7:
        gaps.append("DSA score is below the target of 7/10.")
        recommendations.append("Practice arrays, strings, stacks, queues, trees, and sorting.")
    if profile["aptitude_score"] < 70:
        gaps.append("Aptitude score is below 70/100.")
        recommendations.append("Practice quantitative aptitude and logical reasoning.")
    if profile["communication_skills"] < 6:
        gaps.append("Communication skills need improvement.")
        recommendations.append("Practice mock interviews and short technical explanations.")
    if profile["ml_knowledge"] < 6:
        gaps.append("Machine-learning knowledge can be improved.")
        recommendations.append("Build a small ML project and explain the model evaluation.")
    if profile["system_design"] < 5:
        gaps.append("System design score is below 5/10.")
        recommendations.append("Study basic APIs, databases, scalability, and system components.")
    if profile["internships"] == 0:
        gaps.append("No internship experience entered.")
        recommendations.append("Look for internships and document your project contributions.")
    if not gaps:
        recommendations.append("Keep practicing and maintain your current preparation level.")
    return gaps, recommendations


def make_csv_download(df):
    return df.to_csv(index=False).encode("utf-8")


# Sidebar navigation
st.sidebar.title("🎓 Placement Analytics")
page = st.sidebar.radio(
    "Navigate to",
    ["Home", "Student Assessment", "Analytics Dashboard",
     "Placement Prediction", "Readiness Score",
     "Skill Gap & Recommendations", "Model Performance", "Power BI / Tableau Export"],
)
st.sidebar.caption("Save a student assessment to personalize the dashboard.")

try:
    # mtime changes when the dataset file is replaced.
    stamp = DATA_FILE.stat().st_mtime_ns if DATA_FILE.exists() else 0
    df = load_data(stamp)
    models, metrics, best_model_name = train_models(stamp)
except Exception as exc:
    st.error(f"Could not load/train the placement analytics data: {exc}")
    st.info("Check that placement_data.csv and the required packages are present.")
    st.stop()


def require_profile():
    if "profile" not in st.session_state:
        st.warning("Please complete Student Assessment and click Save Profile & Analyze first.")
        st.stop()
    return st.session_state["profile"]


if page == "Home":
    st.title("🎓 AI-Based Placement Readiness Analytics")
    st.write(
        "Assess a student's placement readiness, estimate placement probability, "
        "identify skill gaps, and explore dataset trends."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Students in dataset", f"{len(df):,}")
    c2.metric("Historical placement rate", f"{df[TARGET].mean() * 100:.1f}%")
    c3.metric("Best model", best_model_name)
    st.markdown("### How to use")
    st.markdown(
        "1. Open **Student Assessment** and enter a student profile.\n"
        "2. Click **Save Profile & Analyze**.\n"
        "3. Open **Analytics Dashboard** to see profile-specific metrics and charts.\n"
        "4. Use **Power BI / Tableau Export** to download CSV files for BI reports."
    )

elif page == "Student Assessment":
    st.title("📝 Student Assessment")
    st.write("Change the values and click Save Profile & Analyze. The personalized dashboard will use this saved profile.")
    old = st.session_state.get("profile", {})
    with st.form("student_assessment_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "ME", "EE", "CE", "Chemical"],
                                  index=["CSE", "IT", "ECE", "ME", "EE", "CE", "Chemical"].index(old.get("branch", "CSE")))
            tier = st.selectbox("College tier", ["Tier-1", "Tier-2", "Tier-3"],
                                index=["Tier-1", "Tier-2", "Tier-3"].index(old.get("college_tier", "Tier-2")))
            cgpa = st.number_input("CGPA (5–10)", min_value=5.0, max_value=10.0,
                                   value=float(old.get("cgpa", 7.6)), step=0.1)
            backlogs = st.number_input("Backlogs", min_value=0, max_value=10,
                                       value=int(old.get("backlogs", 0)), step=1)
            aptitude = st.slider("Aptitude score (0–100)", 0, 100,
                                 int(old.get("aptitude_score", 65)))
        with c2:
            coding = st.slider("Coding skills (0–10)", 0.0, 10.0,
                               float(old.get("coding_skills", 6.0)), 0.1)
            dsa = st.slider("DSA score (0–10)", 0.0, 10.0,
                            float(old.get("dsa_score", 6.0)), 0.1)
            comm = st.slider("Communication skills (0–10)", 0.0, 10.0,
                             float(old.get("communication_skills", 6.5)), 0.1)
            ml = st.slider("ML knowledge (0–10)", 0.0, 10.0,
                           float(old.get("ml_knowledge", 5.0)), 0.1)
            system_design = st.slider("System design (0–10)", 0.0, 10.0,
                                      float(old.get("system_design", 4.5)), 0.1)
        with c3:
            internships = st.number_input("Internships", 0, 10, int(old.get("internships", 1)))
            projects = st.number_input("Projects", 0, 20, int(old.get("projects_count", 2)))
            certifications = st.number_input("Certifications", 0, 20, int(old.get("certifications", 1)))
            hackathons = st.number_input("Hackathons", 0, 20, int(old.get("hackathons", 0)))
            open_source = st.number_input("Open-source contributions", 0, 20,
                                          int(old.get("open_source_contributions", 0)))
            extracurriculars = st.number_input("Extracurriculars", 0, 20,
                                               int(old.get("extracurriculars", 1)))
        submitted = st.form_submit_button("Save Profile & Analyze", type="primary")

    if submitted:
        st.session_state["profile"] = {
            "branch": branch, "college_tier": tier, "cgpa": cgpa, "backlogs": backlogs,
            "coding_skills": coding, "dsa_score": dsa, "aptitude_score": aptitude,
            "communication_skills": comm, "ml_knowledge": ml, "system_design": system_design,
            "internships": internships, "projects_count": projects,
            "certifications": certifications, "hackathons": hackathons,
            "open_source_contributions": open_source, "extracurriculars": extracurriculars,
        }
        st.success("Profile saved. Open Analytics Dashboard to see updated results.")
        st.rerun()

elif page == "Analytics Dashboard":
    st.title("📊 Dynamic Placement Analytics Dashboard")
    st.caption("Personal metrics update from the saved assessment. Dataset metrics describe the historical CSV and change only when the dataset or filters change.")
    profile = st.session_state.get("profile")

    if profile:
        readiness_total, components = calculate_readiness(profile)
        profile_df = profile_to_frame(profile)
        probability = float(models[best_model_name].predict_proba(profile_df[FEATURES])[0, 1])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Your readiness score", f"{readiness_total}/100")
        c2.metric("Your predicted placement chance", f"{probability * 100:.1f}%")
        c3.metric("Your CGPA", f"{profile['cgpa']:.2f}")
        c4.metric("Your branch / tier", f"{profile['branch']} / {profile['college_tier']}")
        st.subheader("Your skill profile vs dataset average")
        compare_cols = ["coding_skills", "dsa_score", "communication_skills", "ml_knowledge", "system_design"]
        compare_names = ["Coding", "DSA", "Communication", "ML", "System design"]
        student_values = [profile[x] for x in compare_cols]
        average_values = [float(df[x].mean()) for x in compare_cols]
        chart_df = pd.DataFrame({
            "Skill": compare_names,
            "Your score": student_values,
            "Dataset average": average_values,
        }).set_index("Skill")
        st.bar_chart(chart_df)
        component_df = pd.DataFrame({
            "Category": list(components.keys()),
            "Score": list(components.values()),
        }).set_index("Category")
        st.subheader("Your readiness breakdown")
        st.bar_chart(component_df)
        st.download_button(
            "Download your assessment for Power BI / Tableau (CSV)",
            make_csv_download(pd.DataFrame([{
                **profile, "readiness_score": readiness_total,
                "predicted_placement_probability": round(probability, 4),
            }])),
            file_name="student_assessment_dashboard.csv",
            mime="text/csv",
        )
    else:
        st.info("No saved profile yet. Complete Student Assessment to activate the personalized dashboard.")

    st.divider()
    st.subheader("Historical dataset analytics")
    f1, f2 = st.columns(2)
    with f1:
        branches = ["All"] + sorted(df["branch"].dropna().unique().tolist())
        selected_branch = st.selectbox("Filter by branch", branches, key="dash_branch")
    with f2:
        tiers = ["All"] + sorted(df["college_tier"].dropna().unique().tolist())
        selected_tier = st.selectbox("Filter by college tier", tiers, key="dash_tier")
    filtered = df.copy()
    if selected_branch != "All":
        filtered = filtered[filtered["branch"] == selected_branch]
    if selected_tier != "All":
        filtered = filtered[filtered["college_tier"] == selected_tier]
    if filtered.empty:
        st.warning("No rows match these filters.")
    else:
        placed_df = filtered[filtered["placed"] == 1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Filtered students", f"{len(filtered):,}")
        m2.metric("Placement rate", f"{filtered['placed'].mean() * 100:.1f}%")
        m3.metric("Average CGPA", f"{filtered['cgpa'].mean():.2f}")
        m4.metric("Average salary (placed)", f"{placed_df['salary_package_lpa'].mean():.2f} LPA" if not placed_df.empty else "N/A")

        chart1, chart2 = st.columns(2)
        with chart1:
            st.markdown("**Placement rate by branch (%)**")
            branch_rate = filtered.groupby("branch")["placed"].mean().mul(100).sort_values()
            st.bar_chart(branch_rate)
        with chart2:
            st.markdown("**Average CGPA by placement status**")
            cgpa_by_status = filtered.assign(
                Placement=filtered["placed"].map({0: "Not placed", 1: "Placed"})
            ).groupby("Placement")["cgpa"].mean()
            st.bar_chart(cgpa_by_status)
        st.markdown("**Salary package by college tier (placed students only)**")
        salary_by_tier = placed_df.groupby("college_tier")["salary_package_lpa"].mean()
        st.bar_chart(salary_by_tier)
        st.download_button(
            "Download filtered dataset for Power BI / Tableau (CSV)",
            make_csv_download(filtered),
            file_name="placement_analytics_filtered.csv",
            mime="text/csv",
        )

elif page == "Placement Prediction":
    st.title("🎯 Placement Prediction")
    profile = require_profile()
    prob = float(models[best_model_name].predict_proba(profile_to_frame(profile)[FEATURES])[0, 1])
    c1, c2 = st.columns(2)
    c1.metric("Selected model", best_model_name)
    c2.metric("Predicted placement probability", f"{prob * 100:.1f}%")
    if prob >= 0.70:
        st.success("High estimated placement probability.")
    elif prob >= 0.50:
        st.warning("Moderate estimated placement probability.")
    else:
        st.error("Lower estimated placement probability. Review the skill-gap recommendations.")
    st.caption("This is a model estimate based on synthetic historical data, not a guarantee of a real placement.")
    st.dataframe(pd.DataFrame([profile]), use_container_width=True)

elif page == "Readiness Score":
    st.title("📈 Placement Readiness Score")
    profile = require_profile()
    total, components = calculate_readiness(profile)
    st.metric("Overall readiness score", f"{total}/100")
    st.progress(total / 100)
    st.bar_chart(pd.Series(components, name="Score"))
    st.caption("Weighted rule-based score: academic 30%, technical 30%, aptitude 20%, communication 10%, experience 10%.")

elif page == "Skill Gap & Recommendations":
    st.title("💡 Skill Gap & Recommendations")
    profile = require_profile()
    gaps, recommendations = get_recommendations(profile)
    left, right = st.columns(2)
    with left:
        st.subheader("Areas to improve")
        if gaps:
            for gap in gaps:
                st.warning(gap)
        else:
            st.success("No major gaps detected by the current rules.")
    with right:
        st.subheader("Suggested actions")
        for item in recommendations:
            st.write(f"• {item}")

elif page == "Model Performance":
    st.title("🤖 Model Performance")
    st.caption("Metrics are measured on a held-out test split of the historical dataset.")
    st.dataframe(metrics.style.format("{:.3f}"), use_container_width=True)
    st.subheader("Model comparison")
    st.bar_chart(metrics[["Accuracy", "F1 score", "ROC AUC"]])
    st.write(f"Best model by ROC AUC: **{best_model_name}**")

elif page == "Power BI / Tableau Export":
    st.title("📥 Power BI / Tableau Export")
    st.write("Download CSV files here, then import them into Microsoft Power BI or Tableau. The CSV export is the connection between this Streamlit app and the BI tool.")
    profile = st.session_state.get("profile")
    if profile:
        total, _ = calculate_readiness(profile)
        probability = float(models[best_model_name].predict_proba(profile_to_frame(profile)[FEATURES])[0, 1])
        personal_export = pd.DataFrame([{
            **profile,
            "readiness_score": total,
            "predicted_placement_probability": round(probability, 4),
        }])
        st.download_button(
            "Download current student profile (CSV)",
            make_csv_download(personal_export),
            file_name="student_assessment_dashboard.csv",
            mime="text/csv",
        )
    else:
        st.info("Save a student profile first to export personalized results.")
    st.download_button(
        "Download full placement dataset (CSV)",
        make_csv_download(df),
        file_name="placement_data_for_bi.csv",
        mime="text/csv",
    )
    st.markdown("### Connect in Power BI")
    st.markdown("1. Download a CSV above.\n2. Open Power BI Desktop.\n3. Select **Home → Get data → Text/CSV**.\n4. Choose the downloaded CSV and select **Load**.\n5. Create visuals using fields such as `branch`, `placed`, `cgpa`, `salary_package_lpa`, `readiness_score`, and `predicted_placement_probability`.")
    st.markdown("### Connect in Tableau")
    st.markdown("1. Download a CSV above.\n2. Open Tableau Desktop or Tableau Public.\n3. Under **Connect**, select **Text File**.\n4. Choose the downloaded CSV.\n5. Drag fields onto Rows, Columns, and Marks to build charts.")
    st.warning("Power BI and Tableau are separate applications. This Streamlit project exports data for them; it does not automatically embed a live Power BI/Tableau report. Refresh or re-import the CSV after downloading updated results.")
