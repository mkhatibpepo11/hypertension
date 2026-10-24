"""Educational demo. Not a clinical decision tool."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Hypertension ML Demo", page_icon="📊", layout="wide")
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 1rem;}
    h1 {font-size: 2rem !important;}
    [data-testid="stVerticalBlock"] {gap: 0.65rem;}
</style>
""", unsafe_allow_html=True)
st.title("Hypertension Classification — Educational Demo")
st.warning("Uses synthetic data. Not a medical diagnosis, screening tool, or clinical risk calculator.")
path = ROOT / "models" / "model.joblib"
if not path.exists():
    st.info("Train the model first: `python src/train.py`")
    st.stop()
artifact = joblib.load(path)  # Load only files you trust.
model = artifact["pipeline"]
features = artifact["features"]
st.write("Enter the values you know. Leave unknown numbers blank or select 'I don't know'.")
st.caption("Missing values are filled using training data. These replacements are not actual measurements.")
labels = {
    "age": "Age (years)",
    "sex": "Sex",
    "residence": "Place of residence",
    "bmi": "Body mass index (BMI, kg/m²)",
    "family_history_hypertension": "Do you have a family history of hypertension?",
    "diabetes": "Do you have diabetes?",
    "smoking": "Do you smoke?",
    "alcohol_heavy": "Do you drink alcohol heavily?",
    "physically_active": "Are you physically active?",
    "high_salt_diet": "Is your diet high in salt?",
    "stroke_history": "Have you ever had a stroke?",
    "myocardial_infarction": "Have you ever had a heart attack?",
    "heart_failure": "Do you have heart failure?",
    "total_cholesterol_mg_dl": "Total cholesterol (mg/dL)",
    "ldl_mg_dl": "LDL cholesterol (mg/dL)",
    "hdl_mg_dl": "HDL cholesterol (mg/dL)",
    "creatinine_mg_dl": "Creatinine (mg/dL)",
}
category_labels = {"Male": "Male", "Female": "Female", "Rural": "Rural", "Urban": "Urban"}


def display_choice(value):
    if value is None:
        return "I don't know"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return category_labels.get(value, str(value))


preprocess = model.named_steps["preprocess"]
numeric_features = preprocess.transformers_[0][2]
categorical_features = preprocess.transformers_[1][2]
encoder = preprocess.named_transformers_["categorical"].named_steps["encode"]
category_options = {
    name: categories.tolist()
    for name, categories in zip(categorical_features, encoder.categories_)
}
values = {}


def render_input(name):
    label = labels.get(name, name)
    if name in numeric_features:
        number = st.number_input(
            label, min_value=0.0, value=None, format="%.2f", key=name,
            placeholder="Enter a value if known",
            help="Leave blank if unknown. Use the unit shown in the field label.",
        )
        values[name] = float("nan") if number is None else number
    else:
        choice = st.selectbox(
            label, options=[None, *category_options[name]], key=name,
            format_func=display_choice,
            help="Options come from the model's training data. Select 'I don't know' if unsure.",
        )
        values[name] = float("nan") if choice is None else choice


groups = {
    "Personal details": ["age", "sex", "residence", "bmi"],
    "Lifestyle": ["smoking", "alcohol_heavy", "physically_active", "high_salt_diet"],
    "Health history": ["family_history_hypertension", "diabetes", "stroke_history",
                       "myocardial_infarction", "heart_failure"],
    "Lab measurements": ["total_cholesterol_mg_dl", "ldl_mg_dl", "hdl_mg_dl",
                         "creatinine_mg_dl"],
}
with st.form("prediction"):
    columns = st.columns(4, gap="large")
    for column, (heading, names) in zip(columns, groups.items()):
        with column:
            st.subheader(heading)
            for name in names:
                if name in features:
                    render_input(name)
    # Keep any additional model features accessible if the dataset changes.
    grouped_features = {name for names in groups.values() for name in names}
    extra_features = [name for name in features if name not in grouped_features]
    if extra_features:
        st.subheader("Additional details")
        extra_columns = st.columns(4)
        for index, name in enumerate(extra_features):
            with extra_columns[index % 4]:
                render_input(name)
    submit = st.form_submit_button("Run demo prediction", type="primary")
if submit:
    frame = pd.DataFrame([values], columns=features)
    if frame.isna().all(axis=None):
        st.info("Enter at least one value to run a prediction.")
        st.stop()
    p = float(model.predict_proba(frame)[0, 1])
    score_column, class_column = st.columns(2)
    score_column.metric("Model score for class 1", f"{p:.1%}")
    class_column.metric("Predicted dataset class", "Hypertensive (1)" if p >= artifact["threshold"] else "Normal (0)")
    st.caption("This model score is not a validated personal probability of disease.")
