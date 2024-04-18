import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
import numpy as np
from streamlit_shap import st_shap
from streamlit_gsheets import GSheetsConnection

conn = st.connection("gsheets", type=GSheetsConnection)
df = conn.read()
# Load data and cache it to avoid reloading every time
@st.cache_data
def load_data():
    # Create a connection object using your specified authentication method
    conn = st.connection("gsheets", type=GSheetsConnection)
    # You need to specify the name or ID of your sheet and possibly the range if not the whole sheet
    df = conn.read()
    return df

data = load_data()
feature_columns = ['Hand_Grip', 'Balance', 'CS_5', 'Age', 'Pain', 'Comorbidities', 'Depression', 'Breath', 'Cognition']
feature_descriptions = {
    'Hand_Grip': "Hand grip strength (times)",
    'Balance': "Balance metric score",
    'CS_5': "5 meter walk speed (m/s)",
    'Age': "Age of the individual",
    'Pain': "Pain assessment score",
    'Comorbidities': "Number of comorbid conditions",
    'Depression': "Depression assessment score",
    'Breath': "Breath hold time (seconds)",
    'Cognition': "Cognitive function score"
}
X = data[feature_columns]
y = data['Disability_2020']

# Initialize and train model only if not in session state
if 'model' not in st.session_state or 'explainer' not in st.session_state:
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X, y)
    explainer = shap.Explainer(model, X)
    st.session_state['model'] = model
    st.session_state['explainer'] = explainer

model = st.session_state['model']
explainer = st.session_state['explainer']

# Use placeholders for dynamic parts
prediction_placeholder = st.empty()
shap_placeholder = st.empty()

# Calculate initial prediction and SHAP values using mean of features
input_df = pd.DataFrame([X.mean()])
initial_prediction = model.predict(input_df)
initial_shap_values = explainer(input_df)

# Display initial prediction and SHAP plot
with prediction_placeholder.container():
    # st.write(f"Initial Predicted Disability: {initial_prediction[0]:.3f}")
    st.markdown(f"""
    <style>
        .prediction {{
            font-size: 20px;
            margin-bottom: 20px;
        }}
        .p_num {{
            color: rgb(255, 13, 87);
        }}
    </style>
    <div class="prediction">Initial Predicted Disability: <span class="p_num">{initial_prediction[0]:.3f}</span></div>
    """, unsafe_allow_html=True)
with shap_placeholder.container():
    st_shap(shap.plots.force(explainer.expected_value, initial_shap_values.values[0], input_df.iloc[0]))

# Sidebar form for input
with st.sidebar.form("input_form"):
    inputs = {}
    inputs["Hand_Grip"] = st.number_input(f"{feature_descriptions["Hand_Grip"]}", value=X["Hand_Grip"].mean())
    inputs["Balance"] = st.number_input(f"{feature_descriptions["Balance"]}", value=X["Balance"].mean())
    inputs["CS_5"] = st.number_input(f"{feature_descriptions["CS_5"]}", value=X["CS_5"].mean())
    inputs["Age"] = st.number_input(f"{feature_descriptions["Age"]}", value=X["Age"].mean())
    options = [(1, 'Yes'), (0, 'No')]

    inputs["Pain"] = st.selectbox(
        feature_descriptions["Pain"],
        options=options,
        format_func=lambda x: x[1],
        index=1 if X["Pain"].mean() <= 0.5 else 0)[0]

    inputs["Comorbidities"] = st.selectbox(
        feature_descriptions["Comorbidities"],
        options=options,
        format_func=lambda x: x[1],
        index=1 if X["Comorbidities"].mean() <= 0.5 else 0)[0]

    inputs["Depression"] = st.selectbox(
        feature_descriptions["Depression"],
        options=options,
        format_func=lambda x: x[1],
        index=1 if X["Depression"].mean() <= 0.5 else 0)[0]
    
    
    inputs["Breath"] = st.number_input(f"{feature_descriptions["Breath"]}", value=X["Breath"].mean())
    inputs["Cognition"] = st.number_input(f"{feature_descriptions["Cognition"]}", value=X["Cognition"].mean())
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)
    shap_values = explainer(input_df)

    # Update only the placeholders with new inputs
    with prediction_placeholder.container():
        # st.write(f"Predicted Disability: {prediction[0]:.3f}")
        st.markdown(f"""
            <style>
                .prediction {{
                    font-size: 20px;
                    margin-bottom: 20px;
                }}
                .p_num {{
                    color: rgb(255, 13, 87);
                }}
            </style>
            <div class="prediction">Predicted Disability: <span class="p_num">{prediction[0]:.3f}</span></div>
            """, unsafe_allow_html=True)
    with shap_placeholder.container():
        st_shap(shap.plots.force(explainer.expected_value, shap_values.values[0], input_df.iloc[0]))
