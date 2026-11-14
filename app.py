import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Patient Health Risk Prediction", page_icon="🩺")


@st.cache_resource
def load(name):
    return pickle.load(open(f"model/{name}", "rb"))


power = load("power_transformers.pkl")      # Yeo-Johnson for Age
selected = load("selected_features.pkl")    # the 6 selected features
scaler = load("SelectKBest_scaler.pkl")
model = load("RandomForest_model.pkl")

GROUPS = {"Young": 0, "Adult": 1, "Middle_Age": 2, "Senior": 3}
RISKS = {"Low": 0, "Medium": 1, "High": 2}

st.title("Patient Health Risk Prediction")
st.caption("Random Forest model · 6 selected features")

with st.form("patient"):
    Age = st.slider("Age (years)", 1, 99, 50)                                   # drag left / right
    c1, c2 = st.columns(2)
    Diabetic = c1.text_input("Diabetic (blank if unknown)", placeholder="e.g. Yes")
    Visit_Year = c2.number_input("Visit year (blank if unknown)", min_value=2000, max_value=2100, value=None, placeholder="e.g. 2025")
    Cholesterol = st.select_slider("Cholesterol status", ["normal", "high"])
    Age_Group = st.select_slider("Age group", list(GROUPS), value="Adult")
    Risk_Score = st.select_slider("Risk score", list(RISKS), value="Medium")
    submitted = st.form_submit_button("Run assessment", type="primary", width="stretch")

if submitted:
    Age_t = power["Age"].transform(pd.DataFrame([[float(Age)]], columns=["Age"]))[0][0]
    Diabetic_was_missing = 0 if Diabetic.strip() else 1
    Visit_Year_was_missing = 0 if Visit_Year is not None else 1
    Cholesterol_Status_normal = 1 if Cholesterol == "normal" else 0

    values = [Age_t, Diabetic_was_missing, Visit_Year_was_missing, Cholesterol_Status_normal, GROUPS[Age_Group], RISKS[Risk_Score]]
    data = scaler.transform(pd.DataFrame([values], columns=selected))

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1] * 100

    if prediction == 1:
        st.error(f"### Disease Detected  ·  {probability:.1f}%")
    else:
        st.success(f"### No Disease  ·  {probability:.1f}% chance of disease")
    st.progress(int(probability), text=f"Disease probability {probability:.1f}%")

    entered = [str(Age), Diabetic or "(blank)", Visit_Year or "(blank)", Cholesterol, Age_Group, Risk_Score]
    how = ["Yeo-Johnson transformed",
           "left blank -> missing -> flag = 1" if Diabetic_was_missing else "value given -> not missing -> flag = 0",
           "left blank -> missing -> flag = 1" if Visit_Year_was_missing else "value given -> not missing -> flag = 0",
           f"Cholesterol_Status='{Cholesterol}' -> {Cholesterol_Status_normal} (1 means normal)",
           "ordinal: Young=0, Adult=1, Middle_Age=2, Senior=3", "ordinal: Low=0, Medium=1, High=2"]
    table = pd.DataFrame({"Feature": selected, "You entered": [str(e) for e in entered],
                          "Value sent to model": [f"{v:.4f}" for v in values], "How it was used": how})
    st.subheader("Input vs value sent to model")
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Practice model (accuracy ≈ 49%, close to chance). Not medical advice.")
