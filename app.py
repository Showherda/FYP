import streamlit as st
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
from groq import Groq
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import numpy as np

# Load the environment variables
load_dotenv()

# Initialize the Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
groq_model = "llama-3.3-70b-versatile"

# Initialize prompts
system_prompt = """
    You are a financial analyst at a large investment firm. Your firm is looking to invest in a company that has a strong ESG (Environmental, Social, and Governance) profile.
    The data scientists in your company have come up with a model that predicts the ESG risk score of a company based on various ESG factors.
    The model takes the following inputs:
    - Issue Green Bonds: Whether the company has issued green bonds (0 or 1)
    - Dow Jones Sustainability Index: Whether the company is part of the Dow Jones Sustainability Index (0 or 1)
    - Recycling: Whether the company has recycling programs (0 or 1)
    - Energy Conservation: Whether the company has energy conservation programs (0 or 1)
    - Climate Change Policy: Whether the company has a climate change policy (0 or 1)
    - Water Treatment: Whether the company has water treatment programs (0 or 1)
    - Biodiversity: Whether the company has biodiversity programs (0 or 1)
    - CSR Commitee: Whether the company has a CSR committee (0 or 1)
    - Waste Management: Whether the company has waste management programs (0 or 1)
    - Net Zero Targets: Whether the company has net zero targets (0 or 1)
    - Disclosure of R&D: Whether the company discloses its R&D activities (0 or 1)
    - Planning Zero Carbon: Whether the company is planning for zero carbon emissions (0 or 1)
    - Government Collaboration: Whether the company collaborates with the government (0 or 1)
    - Renewal Energy: Whether the company uses renewal energy (0 or 1)
    - Environmental Scores: The environmental scores of the company
    - Controversy Level: The controversy level of the company
    - Indirect Carbon Emissions: The indirect carbon emissions of the company
    - Direct Carbon Emissions: The direct carbon emissions of the company
    - Disclosure Scores: The disclosure scores of the company
    - MSCI ESG Index: The MSCI ESG index of the company (0, 1, 2, 3)
    - GHG Sales: The GHG sales of the company
    - SDG Targets: The SDG targets of the company (E, EG, ES, ESG, G, S, SG)
    - Total Carbon Emissions: The total carbon emissions of the company
    - CSR Board Size: The CSR board size of the company
    - ESG Funds: The ESG funds of the company
    - Scope3 Carbon Emissions: The scope3 carbon emissions of the company
    The data scientists have also calculated the following average treatment effects (ATE) for features:
    ATE for features with binary values: {'Issue_green_bonds': {0: 0.11, 1: -0.6}, 'Recycling': {0: -0.52, 1: 0.02}, 'Energy_conservation': {0: 0.02, 1: -0.02}, 'Climate_change_policy': {0: 0.08, 1: -0.11}, 'Water_treatment': {0: 0.03, 1: -0.1}, 'Biodiversity': {0: 0.16, 1: -0.61}, 'CSR_commitee': {0: -0.02, 1: 0.01}, 'Waste_management': {0: 0.09, 1: 0.33}, 'Net_zero_targets': {0: 0.36, 1: -0.01}, 'Disclosure_of_R&D': {0: -0.42, 1: 0.78}, 'Planning_zero_carbon': {0: 0.04, 1: -0.0}, 'Govt_collaboration': {0: 0.07, 1: -0.06}, 'Renewal_energy': {0: -0.45, 1: 0.23}}
    ATE for features with continuous values: {'Indirect_carbon_emissions': {(2, 1): -0.0, (3, 1): 0.18, (4, 1): 0.38, (1, 2): 0.02, (3, 2): 0.13, (4, 2): 0.34, (1, 3): 0.01, (2, 3): -0.15, (4, 3): 0.07, (1, 4): -0.19, (2, 4): -0.24, (3, 4): -0.13}, 'Direct_carbon_emissions': {(2, 1): 0.15, (3, 1): 0.04, (4, 1): -0.12, (1, 2): -0.04, (3, 2): -0.1, (4, 2): -0.13, (1, 3): 0.02, (2, 3): -0.36, (4, 3): -0.13, (1, 4): 0.18, (2, 4): -1.16, (3, 4): -0.57}, 'GHG_sales': {(2, 1): -1.02, (3, 1): -3.44, (4, 1): -6.98, (1, 2): -1.16, (3, 2): -3.94, (4, 2): -6.49, (1, 3): 2.82, (2, 3): 2.88, (4, 3): -0.75, (1, 4): 4.05, (2, 4): 3.56, (3, 4): 0.67}, 'CSR_board_size': {(1, 0): None, (3, 1): -0.08, (4, 1): -0.09, (1, 2): None, (3, 2): 0.49, (4, 2): -0.04, (1, 3): None, (4, 3): -0.12, (1, 4): 0.55, (3, 4): 0.5}, 'ESG_funds': {(2, 1): 0.34, (3, 1): 1.31, (4, 1): 2.15, (1, 2): 6.1, (3, 2): 1.39, (4, 2): 2.66, (1, 3): 7.61, (2, 3): -1.29, (4, 3): 1.61, (1, 4): 6.14, (2, 4): -4.47, (3, 4): -1.73}, 'Total_carbon_emissions': {(2, 1): -0.17, (3, 1): -0.12, (4, 1): 0.06, (1, 2): 0.64, (3, 2): -0.04, (4, 2): -0.01, (1, 3): 0.74, (2, 3): 0.02, (4, 3): 0.02, (1, 4): 0.68, (2, 4): 0.02, (3, 4): 0.11}, 'Scope3_carbon_emissions': {(1, 0): None, (3, 1): 0.25, (4, 1): 0.31, (1, 2): None, (3, 2): 0.09, (4, 2): 0.94, (1, 3): None, (4, 3): 0.07, (1, 4): -0.29, (3, 4): -0.04}}
    The rest of the features are scores calculated by other companies which aren't something to be experimented on.
    ATE are basically potential outcomes.
    For binary values, ATE[feature][0] is the potential outcome if the feature value was originally 0 and it was changed to 1. ATE[feature][1] is the potential outcome if the feature value was originally 1 and it was changed to 0.
    For continuous values, ATE[feature][(j, i)] is the potential value for the feature if the original value was in the j-th quartile (for that feature) and it was changed to the mean of the i-th quartile (for the same feature).
    The user inputs the values of the features and the model predicts the ESG risk score of the company.
    You will be given the user input and the predicted ESG risk score (predicted by the model developed by the data scientists).
    You have to suggest if the company is good for investing and how to improve the ESG risk score of the company based on the user input.
    Do not include text like "here is your output" or "here is the prediction" in your response. Respond like how a human will answer a question.
    Do not make your suggestion too technical. Keep it simple and easy to understand. Do not mention any values or numbers in your suggestion.
"""

# Load the trained random forest regressor model
model = joblib.load('models/model.pkl')

# Load the dataset
df = pd.read_csv('data/esgdata_cleaned.csv')

# Define the input fields
st.title("ESG Risk Score Prediction")

# Create two columns
col1, col2 = st.columns(2)

# Define mapping dictionaries
SDG_target_mapping = {'E': 0, 'EG': 1, 'ES': 2, 'ESG': 3, 'G': 4, 'S': 5, 'SG': 6}
binary_features = ['Issue_green_bonds', 'Dow_jones_sustainability_index', 'Recycling', 'Energy_conservation', 'Climate_change_policy', 'Water_treatment', 'Biodiversity', 'CSR_commitee', 'Waste_management', 'Net_zero_targets', 'Disclosure_of_R&D', 'Planning_zero_carbon', 'Govt_collaboration', 'Renewal_energy']
binary_mapping = {'No': 0, 'Yes': 1}

with col1:
    Issue_green_bonds = binary_mapping[st.selectbox("Issue Green Bonds", ["No", "Yes"])]
    Dow_jones_sustainability_index = binary_mapping[st.selectbox("Dow Jones Sustainability Index", ["No", "Yes"])]
    Recycling = binary_mapping[st.selectbox("Recycling", ["No", "Yes"])]
    Energy_conservation = binary_mapping[st.selectbox("Energy Conservation", ["No", "Yes"])]
    Climate_change_policy = binary_mapping[st.selectbox("Climate Change Policy", ["No", "Yes"])]
    Water_treatment = binary_mapping[st.selectbox("Water Treatment", ["No", "Yes"])]
    Biodiversity = binary_mapping[st.selectbox("Biodiversity", ["No", "Yes"])]
    CSR_commitee = binary_mapping[st.selectbox("CSR Commitee", ["No", "Yes"])]
    Waste_management = binary_mapping[st.selectbox("Waste Management", ["No", "Yes"])]
    Net_zero_targets = binary_mapping[st.selectbox("Net Zero Targets", ["No", "Yes"])]
    Disclosure_of_RnD = binary_mapping[st.selectbox("Disclosure of R&D", ["No", "Yes"])]
    Planning_zero_carbon = binary_mapping[st.selectbox("Planning Zero Carbon", ["No", "Yes"])]
    Govt_collaboration = binary_mapping[st.selectbox("Government Collaboration", ["No", "Yes"])]
    Renewal_energy = binary_mapping[st.selectbox("Renewal Energy", ["No", "Yes"])]

with col2:
    Environmental_scores = st.number_input("Environmental Scores", min_value=0.0)
    Controversy_level = st.number_input("Controversy Level", min_value=0.0)
    Indirect_carbon_emissions = st.number_input("Indirect Carbon Emissions")
    Direct_carbon_emissions = st.number_input("Direct Carbon Emissions")
    Disclosure_scores = st.number_input("Disclosure Scores", min_value=0.0)
    MSCI_ESG_index = st.selectbox("MSCI ESG Index", [0, 1, 2, 3])
    GHG_sales = st.number_input("GHG Sales", min_value=0.0)
    SDG_targets = SDG_target_mapping[st.selectbox("SDG Targets", SDG_target_mapping)]
    Total_carbon_emissions = st.number_input("Total Carbon Emissions")
    CSR_board_size = st.number_input("CSR Board Size", step=1)
    ESG_funds = st.number_input("ESG Funds", min_value=0.0)
    Scope3_carbon_emissions = st.number_input("Scope3 Carbon Emissions")

user_input = {
    'Disclosure_of_R&D': [Disclosure_of_RnD],
    'Dow_jones_sustainability_index': [Dow_jones_sustainability_index],
    'Controversy_level': [Controversy_level],
    'Waste_management': [Waste_management],
    'Planning_zero_carbon': [Planning_zero_carbon],
    'ESG_funds': [ESG_funds],
    'Indirect_carbon_emissions': [Indirect_carbon_emissions],
    'Water_treatment': [Water_treatment],
    'Recycling': [Recycling],
    'Govt_collaboration': [Govt_collaboration],
    'Energy_conservation': [Energy_conservation],
    'Scope3_carbon_emissions': [Scope3_carbon_emissions],
    'CSR_board_size': [CSR_board_size],
    'Disclosure_scores': [Disclosure_scores],
    'Direct_carbon_emissions': [Direct_carbon_emissions],
    'Issue_green_bonds': [Issue_green_bonds],
    'Total_carbon_emissions': [Total_carbon_emissions],
    'CSR_commitee': [CSR_commitee],
    'Net_zero_targets': [Net_zero_targets],
    'Biodiversity': [Biodiversity],
    'Renewal_energy': [Renewal_energy],
    'Climate_change_policy': [Climate_change_policy],
    'MSCI_ESG_index': [MSCI_ESG_index],
    'GHG_sales': [GHG_sales],
    'SDG_targets': [SDG_targets],
    'Environmental_scores': [Environmental_scores]
}

expected_features = model.feature_names_in_
user_input_ordered = {feature: user_input[feature] for feature in expected_features}

def get_prediction_and_suggestion():
    xgb_prediction = round(model.predict(pd.DataFrame(user_input_ordered))[0], 4)
    llama_suggestion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"User input: {user_input_ordered}. Predicted ESG risk score: {xgb_prediction}"
            }
        ],
        model=groq_model,
    )
    llama_suggestion = llama_suggestion.choices[0].message.content
    return xgb_prediction, llama_suggestion

# Prediction section
if st.button("Predict"):
    with st.spinner("Calculating..."):
        xgb_prediction, llama_suggestion = get_prediction_and_suggestion()
        st.session_state.prediction = xgb_prediction
        st.session_state.llama_suggestion = llama_suggestion

if 'prediction' in st.session_state:
    st.write(f"Predicted ESG Risk Score: {st.session_state.prediction:.4f}")
    st.write(f"Suggestion: {st.session_state.llama_suggestion}")

# Partial Dependence Plot section
st.title("Partial Dependence Plot")

feature_to_plot = st.selectbox("Select feature for PDP", expected_features)

if st.button("Generate PDP"):
    with st.spinner("Generating..."):
        fig, ax = plt.subplots(figsize=(10, 6))
        if feature_to_plot == 'SDG_targets':
            PartialDependenceDisplay.from_estimator(model, df[expected_features], [feature_to_plot], ax=ax)
            ax.set_xticks(list(SDG_target_mapping.values()))
            ax.set_xticklabels(list(SDG_target_mapping.keys()), rotation=45, ha='right')
        elif feature_to_plot in binary_features:
            PartialDependenceDisplay.from_estimator(model, df[expected_features], [feature_to_plot], ax=ax)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['No', 'Yes'])
        else:
            PartialDependenceDisplay.from_estimator(model, df[expected_features], [feature_to_plot], ax=ax)
        fig.tight_layout()
        st.session_state.pdp_fig = fig

if 'pdp_fig' in st.session_state:
    st.pyplot(st.session_state.pdp_fig)