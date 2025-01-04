import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained random forest regressor model
model = joblib.load('models/random_forest_model.pkl')

# Define the input fields
st.title("ESG Risk Score Prediction")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    Issue_green_bonds = st.selectbox("Issue Green Bonds", [0, 1])
    Dow_jones_sustainability_index = st.selectbox("Dow Jones Sustainability Index", [0, 1])
    Recycling = st.selectbox("Recycling", [0, 1])
    Energy_conservation = st.selectbox("Energy Conservation", [0, 1])
    Climate_change_policy = st.selectbox("Climate Change Policy", [0, 1])
    Water_treatment = st.selectbox("Water Treatment", [0, 1])
    Biodiversity = st.selectbox("Biodiversity", [0, 1])
    CSR_commitee = st.selectbox("CSR Commitee", [0, 1])
    Waste_management = st.selectbox("Waste Management", [0, 1])
    Net_zero_targets = st.selectbox("Net Zero Targets", [0, 1])
    Disclosure_of_RnD = st.selectbox("Disclosure of R&D", [0, 1])
    Planning_zero_carbon = st.selectbox("Planning Zero Carbon", [0, 1])
    Govt_collaboration = st.selectbox("Government Collaboration", [0, 1])
    Renewal_energy = st.selectbox("Renewal Energy", [0, 1])

with col2:
    Environmental_scores = st.number_input("Environmental Scores")
    Controversy_level = st.number_input("Controversy Level")
    Indirect_carbon_emissions = st.number_input("Indirect Carbon Emissions")
    Direct_carbon_emissions = st.number_input("Direct Carbon Emissions")
    Disclosure_scores = st.number_input("Disclosure Scores")
    MSCI_ESG_index = st.selectbox("MSCI ESG Index", [0, 1, 2, 3])
    GHG_sales = st.number_input("GHG Sales")
    SDG_targets = st.selectbox("SDG Targets", {'E': 0, 'EG': 1, 'ES': 2, 'ESG': 3, 'G': 4, 'S': 5, 'SG': 6})
    Total_carbon_emissions = st.number_input("Total Carbon Emissions")
    CSR_board_size = st.number_input("CSR Board Size")
    ESG_funds = st.number_input("ESG Funds")
    Scope3_carbon_emissions = st.number_input("Scope3 Carbon Emissions")

# Create a DataFrame for the input
input_data = pd.DataFrame({
    'Issue_green_bonds': [Issue_green_bonds],
    'Dow_jones_sustainability_index': [Dow_jones_sustainability_index],
    'Environmental_scores': [Environmental_scores],
    'Controversy_level': [Controversy_level],
    'Recycling': [Recycling],
    'Indirect_carbon_emissions': [Indirect_carbon_emissions],
    'Energy_conservation': [Energy_conservation],
    'Climate_change_policy': [Climate_change_policy],
    'Direct_carbon_emissions': [Direct_carbon_emissions],
    'Water_treatment': [Water_treatment],
    'Biodiversity': [Biodiversity],
    'CSR_commitee': [CSR_commitee],
    'Disclosure_scores': [Disclosure_scores],
    'MSCI_ESG_index': [MSCI_ESG_index],
    'GHG_sales': [GHG_sales],
    'Waste_management': [Waste_management],
    'Net_zero_targets': [Net_zero_targets],
    'Disclosure_of_R&D': [Disclosure_of_RnD],
    'Planning_zero_carbon': [Planning_zero_carbon],
    'Govt_collaboration': [Govt_collaboration],
    'SDG_targets': [SDG_targets],
    'Total_carbon_emissions': [Total_carbon_emissions],
    'CSR_board_size': [CSR_board_size],
    'ESG_funds': [ESG_funds],
    'Scope3_carbon_emissions': [Scope3_carbon_emissions],
    'Renewal_energy': [Renewal_energy]
})

# Predict the risk score
if st.button("Predict Risk Score"):
    prediction = model.predict(input_data)
    st.write(f"Predicted Risk Score: {prediction[0]}")