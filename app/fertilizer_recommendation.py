import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import os
from datetime import datetime

# --- Configuration and Setup ---

# Define the relative path to the models directory
# Adjust this path if your 'models' directory is structured differently
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../models') 
LATEST_MODEL_PATH = os.path.join(MODELS_DIR, 'fertilizer_recommendation_model_latest.joblib')
LATEST_METADATA_PATH = os.path.join(MODELS_DIR, 'fertilizer_model_metadata_latest.json')

# Original categorical data (used for UI dropdowns and to map the encoded target)
# NOTE: This assumes the LabelEncoder assigned labels alphabetically or by first appearance
SOIL_TYPES = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
CROP_TYPES = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 
              'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']
# The order from your 'Fertilizer Name' (y_test) LabelEncoder mapping
FERTILIZER_MAP = {
    0: '14-35-14',
    1: '28-28',
    2: 'DAP',
    3: 'MOP',
    4: 'Potash',
    5: 'SSP',
    6: 'Urea'
}


@st.cache_resource
def load_model_and_metadata():
    """Loads the trained model pipeline and metadata."""
    try:
        # Load the model
        model = joblib.load(LATEST_MODEL_PATH)
        
        # Load metadata
        with open(LATEST_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
            
        st.success("Model and Metadata loaded successfully! ✅")
        return model, metadata
    except FileNotFoundError as e:
        st.error(f"Error: Model or metadata file not found. Please ensure your files are in the correct path: {MODELS_DIR}")
        st.code(f"Missing file: {e.filename}", language='bash')
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during loading: {e}")
        st.stop()

# Load the resources
model_pipeline, metadata = load_model_and_metadata()

# Get the exact feature order from the metadata for safe prediction
try:
    FEATURE_ORDER = metadata['feature_info']['feature_columns']
except KeyError:
    # Fallback/Debug feature order based on common sense and the error message
    st.warning("Could not find feature order in metadata. Using assumed order.")
    FEATURE_ORDER = ['Temparature', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Phosphorous', 'Potassium', 'pH', 'Humidity ']


# --- Streamlit App Interface ---

st.set_page_config(
    page_title="Fertilizer Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌱 Fertilizer Recommendation System")
st.markdown("Enter the soil and crop characteristics below to get a fertilizer recommendation.")

# --------------------------------------------------------------------------------



# --------------------------------------------------------------------------------

## User Input Form

with st.form("recommendation_form"):
    st.header("Input Features")
    col1, col2, col3 = st.columns(3)
    
    # Column 1: pH, Moisture, Temperature, Humidity
    with col1:
        st.subheader("Soil Conditions (Numerical)")
        # CRITICAL: Use correct feature name 'Temparature' in the prediction data later
        temp = st.number_input("Temperature (°C)", min_value=1.0, max_value=60.0, value=25.0, step=0.1)
        # CRITICAL: Added the missing 'Humidity' input
        humidity = st.number_input("Humidity (%)", min_value=1.0, max_value=100.0, value=60.0, step=0.1) 
        moisture = st.number_input("Moisture (%)", min_value=1.0, max_value=100.0, value=30.0, step=0.1)
        
    # Column 2: Nutrients (NPK)
    with col2:
        st.subheader("Nutrient Content (PPM)")
        nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=80, step=1)
        # CRITICAL: Use correct feature name 'Phosphorous' in the prediction data later
        phosphorous_input = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50, step=1)
        potassium = st.number_input("Potassium (K)", min_value=0, max_value=200, value=40, step=1)
        ph = st.number_input("pH Value", min_value=3.0, max_value=10.0, value=6.5, step=0.01)

    # Column 3: Categorical Features
    with col3:
        st.subheader("Categorical Factors")
        soil_type = st.selectbox("Soil Type", options=SOIL_TYPES)
        crop_type = st.selectbox("Crop Type", options=CROP_TYPES)
        
        # Get the Label Encoded indices
        soil_type_encoded = SOIL_TYPES.index(soil_type)
        crop_type_encoded = CROP_TYPES.index(crop_type)

    st.markdown("---")
    submitted = st.form_submit_button("Get Recommendation", type="primary")

# --------------------------------------------------------------------------------

## Prediction Logic and Output

if submitted:
    
    # 1. Create a dictionary with the EXACT column names from the training data
    data = {
        # FIXES: Match the exact column names the model was trained with
        'Temparature': [temp],          # Match the typo 'Temparature'
        'Moisture': [moisture],
        'Soil Type': [soil_type_encoded],
        'Crop Type': [crop_type_encoded],
        'Nitrogen': [nitrogen],
        'Phosphorous': [phosphorous_input], # Match the spelling 'Phosphorous'
        'Potassium': [potassium],
        'pH': [ph],
        'Humidity ': [humidity]         # Match the trailing space 'Humidity '
    }
    
    # Create the DataFrame and re-order columns
    input_df = pd.DataFrame(data)
    
    # CRITICAL: Re-order the columns to match the exact training order for the ColumnTransformer
    input_df = input_df[FEATURE_ORDER] 

    st.subheader("Prediction Input (Model View)")
    st.dataframe(input_df, hide_index=True)
    
    try:
        # 2. Make the prediction using the full pipeline
        prediction_encoded = model_pipeline.predict(input_df)
        
        # 3. Decode the prediction
        predicted_label_index = prediction_encoded[0]
        recommended_fertilizer = FERTILIZER_MAP.get(predicted_label_index, "Unknown Fertilizer")

        # 4. Display the result
        st.markdown("## 🎯 Recommended Fertilizer")
        st.success(f"Based on the provided inputs, the best fertilizer for **{crop_type}** in **{soil_type}** soil is:")
        st.balloons()
        
        st.markdown(f"""
            <div style="background-color:#E8F8F5; padding: 25px; border-radius: 12px; border-left: 6px solid #00A36C; text-align: center;">
                <h1 style="color:#00A36C; margin: 0; font-size: 48px;">{recommended_fertilizer}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please check the model file integrity and ensure the feature inputs match the expected format.")