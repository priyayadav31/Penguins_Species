import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler (save them as pickle)
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
#Loads your trained ML model from best_model.pkl.
#Loads the scaler used during training from scaler.pkl.
#This allows the app to predict without retraining.

# Displays the title at the top of the web app.
st.title("Penguin Species Prediction")

# Input features
bill_length = st.number_input("Bill Length (mm)")
bill_depth = st.number_input("Bill Depth (mm)")
flipper_length = st.number_input("Flipper Length (mm)")
body_mass = st.number_input("Body Mass (g)")
sex = st.selectbox("Sex", ["Male", "Female"])
island = st.selectbox("Island", ["Biscoe", "Dream", "Torgersen"])
# number_input → lets the user enter numeric features (bill length, depth, etc.).
# selectbox → lets the user select categorical features (sex, island).

# Map categorical to numbers
sex_map = {"Male": 1, "Female": 0}
island_map = {"Biscoe": 0, "Dream": 1, "Torgersen": 2}
# ML models only understand numbers, not text.
# Converts Male/Female and Island names to numeric codes.


# Prediction
if st.button("Predict"):
    features = np.array([[bill_length, bill_depth, flipper_length, body_mass, sex_map[sex], island_map[island]]])
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)
    species_map = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
    st.success(f"Predicted Species: {species_map[pred[0]]}")
##When the user clicks the Predict button:

#Collect features into a 2D array:
#[[bill_length, bill_depth, flipper_length, body_mass, sex, island]]

#Scale features using the previously loaded scaler.

#Standardizes the values like during training.

#Predict species using the trained model.

#Convert numeric prediction back to species name (Adelie, Chinstrap, Gentoo).

#Show result on the app with st.success().'''