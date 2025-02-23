import streamlit as st
import pandas as pd
import xgboost as xgb

# from sklearn.model_selection import train_test_split
import numpy as np
import shap
from streamlit_shap import st_shap
from streamlit_gsheets import GSheetsConnection
import time

# Load data and cache it to avoid reloading every time
@st.cache_data(persist="disk")
def load_data():
    # Create a connection object using your specified authentication method
    print("load_data")
    conn = st.connection("gsheets", type=GSheetsConnection)
    # You need to specify the name or ID of your sheet and possibly the range if not the whole sheet
    df = conn.read()
    return df

feature_columns = ['Pain', 'Breath','Age', 'Hand_Grip','CS_5','Cognition','Comorbidities','Depression','Balance']
feature_descriptions = {
    'Hand_Grip': "Hand Grip",
    'Balance': "Standing Balance",
    'CS_5': "Five-repetition Chair Stand Test (CS-5)",
    'Age': "Age",
    'Pain': "Pain",
    'Comorbidities': "Number of Comorbidities",
    'Depression': "Depression",
    'Breath': "Respiratory Function",
    'Cognition': "Cognitive Function"
}

@st.cache_data(persist="disk")
def load_model_explainer():
    print("load_model")
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X, y)
    explainer = shap.Explainer(model, X)
    input_df = pd.DataFrame([X.mean()])
    prediction = model.predict(input_df)
    shap_values = explainer(input_df)
    return [model, explainer,input_df,prediction,shap_values]


data = load_data()

X = data[feature_columns]
y = data['Disability_2020']

# Initialize and train model only if not in session state
# if 'model' not in st.session_state or 'explainer' not in st.session_state:
#     print("model new")
#     model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
#     model.fit(X, y)
#     explainer = shap.Explainer(model, X)
#     st.session_state['model'] = model
#     st.session_state['explainer'] = explainer

# model = st.session_state['model']
# explainer = st.session_state['explainer']
[model, explainer,input_df,prediction,shap_values] = load_model_explainer()

# Use placeholders for dynamic parts
prediction_placeholder = st.empty()
shap_placeholder = st.empty()

# Sidebar form for input
# Sidebar form for input
with st.sidebar.form("input_form"):
    inputs = {}
    options = [(1, 'Yes'), (0, 'No')]
    boptions = [(0, '0'), (1, '1'), (2, '2'),(3, '3')]
    coptions = [(0, '0'), (1, '1'),(2, '≥ 2')]
    comorbidity_average = X["Comorbidities"].mean()

    # Determine the default index based on the average value
    if comorbidity_average < 0.5:
        com_default_index = 0  # '0'
    elif comorbidity_average < 1.5:
        com_default_index = 1  # '1'
    else:
        com_default_index = 2  # '≥ 2'

    ba_default_index = int(round(X["Balance"].mean()))

    if ba_default_index >= len(boptions):
        ba_default_index = len(boptions) - 1  # Adjust if the rounded mean is out of the option range


    inputs["Pain"] = st.selectbox(
        feature_descriptions["Pain"],
        options=options,
        format_func=lambda x: x[1],
        index=1 if X["Pain"].mean() <= 0.5 else 0)[0]
    
    inputs["Breath"] = st.number_input(
        label=feature_descriptions["Breath"],
        value=int(round(X["Breath"].mean())),  
        step=1  
    )

    inputs["Age"] = st.number_input(
        label=feature_descriptions["Age"],
        value=int(round(X["Age"].mean())),  
        step=1  
    )

    inputs["Hand_Grip"] = st.number_input(
    feature_descriptions["Hand_Grip"], 
    value=X["Hand_Grip"].mean())

    inputs["CS_5"] = st.number_input(
    feature_descriptions["CS_5"], 
    value=X["CS_5"].mean())
    
    inputs["Cognition"] = st.number_input(
        label=feature_descriptions["Cognition"],
        value=int(round(X["Cognition"].mean())),  
        step=1  
    )

    inputs["Comorbidities"] = st.selectbox(
    feature_descriptions["Comorbidities"],
    options=coptions,
    format_func=lambda x: x[1],
    index=com_default_index)[0]

    inputs["Depression"] = st.selectbox(
        feature_descriptions["Depression"],
        options=options,
        format_func=lambda x: x[1],
        index=1 if X["Depression"].mean() <= 0.5 else 0)[0]
    
    inputs["Balance"] = st.selectbox(
        feature_descriptions["Balance"],
        options=boptions,
        format_func=lambda x: x[1],
        index=ba_default_index)[0]



    submitted = st.form_submit_button("Predict")



# Display initial prediction and SHAP plot
with prediction_placeholder.container():
    # st.write(f"Initial Predicted probability of disability: {initial_prediction[0]:.3f}")
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
    <div class="prediction">Predicted probability of disability: <span class="p_num">{prediction[0] * 100:.1f}%</span></div>
    """, unsafe_allow_html=True)
with shap_placeholder.container():
    st_shap(shap.plots.force(explainer.expected_value, shap_values.values[0], input_df.iloc[0]))

if submitted:
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)
    shap_values = explainer(input_df)

    # Update only the placeholders with new inputs
    with prediction_placeholder.container():
        # st.write(f"Predicted probability of disability: {prediction[0]:.3f}")
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
            <div class="prediction">Predicted probability of disability: <span class="p_num">{prediction[0] * 100:.1f}%</span></div>
            """, unsafe_allow_html=True)
    with shap_placeholder.container():
        st_shap(shap.plots.force(explainer.expected_value, shap_values.values[0], input_df.iloc[0]))
