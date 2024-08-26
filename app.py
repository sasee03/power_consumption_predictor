import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from datetime import datetime
import joblib
import numpy as np
import plotly.graph_objects as go

# Set page config - this should be the first Streamlit command
st.set_page_config(page_title="⚡ Tetouan Power Predictor", page_icon="⚡", layout="wide")

# Function to add background image
import base64

def add_bg_from_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Change text color in the main content area to black, except for the sidebar */
        .stApp > div:nth-of-type(1) > div:not(.stSidebar) {{
            color: black;
        }}

        /* Keep the sidebar text its default color */
        .stSidebar, .stSidebar p, .stSidebar label {{
            color: black !important;
        }}

        /* Ensure input and button text remains unchanged */
        .stTextInput, .stButton, .stTimeInput {{
            color: black !important;
        }}

        /* Change color of headers, subheaders, etc., in the main content area */
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with the correct image path
add_bg_from_base64("C:\\Users\\sasee\\OneDrive\\Desktop\\Tau\\84f5d950-08de-420e-abce-9e06a45455fa.png")
import base64
from pathlib import Path

def add_bg_to_sidebar(image_path):
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            st.sidebar.error(f"Image file not found: {image_path}")
            return

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        st.markdown(
            f"""
            <style>
            [data-testid="stSidebar"] {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            [data-testid="stSidebar"] > div:first-child {{
                background-color: rgba(0, 0, 0, 0.5);  /* semi-transparent overlay */
            }}
            [data-testid="stSidebar"] .stMarkdown p, 
            [data-testid="stSidebar"] .stSlider label,
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {{
                color: white !important;
            }}
            .stApp {{
                background-color: transparent;
            }}
            .stApp > header {{
                background-color: transparent;
            }}
            .stTextInput div, .stTextInput label, .stDateInput div, .stTimeInput div, .stSlider div, .stSlider label {{
                color: black !important;
            }}
            .stButton > button {{
                color: white !important;
                background-color: rgba(0, 0, 0, 0.5) !important;
                border: 1px solid white !important;
            }}
            
            .stDateInput > div[data-baseweb="input"] > div,
            .stTimeInput > div[data-baseweb="input"] > div {{
                background-color: rgba(255, 255, 255, 0.1);
                color: white !important;
            }}
            .stTimeInput input, .stTimeInput div[data-baseweb="select"] div {{
            color: white !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.sidebar.error(f"Error setting sidebar background: {str(e)}")

# Call the function with the correct image path for the sidebar
sidebar_bg_path = "C:\\Users\\sasee\\OneDrive\\Desktop\\Tau\\dd17b7d9-c44f-45b7-a81a-dc2b1fdbcb9f.png"
add_bg_to_sidebar(sidebar_bg_path)

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model('model.keras')  # Replace with the actual model path

model = load_trained_model()

# Load the scaler
@st.cache_resource
def load_scaler():
    return joblib.load('scaler_input.joblib')  # Replace with the actual path

scaler_input = load_scaler()

# Function to create features
def create_features(input_df):
    try:
        input_df['hour'] = input_df.index.hour
        input_df['minute'] = input_df.index.minute
        input_df['dayofweek'] = input_df.index.dayofweek
        input_df['quarter'] = input_df.index.quarter
        input_df['month'] = input_df.index.month
        input_df['day'] = input_df.index.day
        input_df['year'] = input_df.index.year
        input_df['season'] = input_df['month'] % 12 // 3 + 1
        input_df['dayofyear'] = input_df.index.dayofyear
        input_df['dayofmonth'] = input_df.index.day
        input_df['weekofyear'] = input_df.index.isocalendar().week

        input_df['is_weekend'] = input_df['dayofweek'].isin([5, 6]).astype(int)
        input_df['is_month_start'] = (input_df['dayofmonth'] == 1).astype(int)
        input_df['is_month_end'] = (input_df['dayofmonth'] == input_df.index.days_in_month).astype(int)
        input_df['is_quarter_start'] = ((input_df['dayofmonth'] == 1) & (input_df['month'] % 3 == 1)).astype(int)
        input_df['is_quarter_end'] = (input_df['dayofmonth'] == input_df.groupby(['year', 'quarter'])['dayofmonth'].transform('max')).astype(int)

        input_df['is_working_day'] = input_df['dayofweek'].isin([0, 1, 2, 3, 4]).astype(int)
        input_df['is_business_hours'] = input_df['hour'].between(9, 17).astype(int)
        input_df['is_peak_hour'] = input_df['hour'].isin([8, 12, 18]).astype(int)

        input_df['minute_of_day'] = input_df['hour'] * 60 + input_df['minute']
        input_df['minute_of_week'] = (input_df['dayofweek'] * 24 * 60) + input_df['minute_of_day']

        return input_df.astype(float)
    except Exception as e:
        st.error(f"Error in create_features: {str(e)}")
        return None

# Function to create gauge chart
def create_gauge_chart(value, title, max_value):
    return go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_value*0.33], 'color': "lightgreen"},
                {'range': [max_value*0.33, max_value*0.66], 'color': "yellow"},
                {'range': [max_value*0.66, max_value], 'color': "red"}
            ],
        }
    ))

# Main function to update predictions and charts
def update_predictions(date_input, time_input, temperature, humidity, wind_speed, general_diffuse_flows, diffuse_flows):
    try:
        selected_datetime = datetime.combine(date_input, time_input)
        
        input_data = pd.DataFrame({
            'Temperature': [temperature],
            'Humidity': [humidity],
            'Wind Speed': [wind_speed],
            'general diffuse flows': [general_diffuse_flows],
            'diffuse flows': [diffuse_flows]
        }, index=[selected_datetime])

        # Check for NaN values
        if input_data.isnull().values.any():
            st.error("Error: Input data contains NaN values. Please check your inputs.")
            return

        input_data = create_features(input_data)
        
        # Check for NaN values again after feature creation
        if input_data is None or input_data.isnull().values.any():
            st.error("Error: Feature creation resulted in NaN values. Please check your inputs and feature creation function.")
            return

        scaled_input_data = scaler_input.transform(input_data)
        prediction = model.predict(scaled_input_data)
        prediction = np.maximum(prediction, 0)

        max_value = max(prediction[0]) * 1.2

        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(create_gauge_chart(prediction[0][0], "Zone 1 (kWh)", max_value), use_container_width=True)
        with col2:
            st.plotly_chart(create_gauge_chart(prediction[0][1], "Zone 2 (kWh)", max_value), use_container_width=True)
        with col3:
            st.plotly_chart(create_gauge_chart(prediction[0][2], "Zone 3 (kWh)", max_value), use_container_width=True)

        total_consumption = sum(prediction[0])
        st.subheader(f"🏙️ Total City Consumption: {total_consumption:.2f} kWh")

        zone_data = pd.DataFrame({
            'Zone': ['Zone 1', 'Zone 2', 'Zone 3'],
            'Consumption': prediction[0]
        })
        st.bar_chart(zone_data.set_index('Zone'))

        with st.expander("🔍 View Input Data"):
            st.dataframe(input_data)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your inputs and try again.")

# Streamlit app

st.title("⚡ Tetouan City Power Consumption Predictor")
st.markdown("Predict power consumption for Tetouan City's three zones! 🏙️🔌")

# Create a form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📅 Date and Time")
        date_input = st.date_input("Select Date 📆")
        time_input = st.time_input("Select Time 🕒")

    with col2:
        st.subheader("🌡️ Weather Conditions")
        temperature = st.slider("Temperature (°C) 🌡️", -10.0, 50.0, value=25.0)
        humidity = st.slider("Humidity (%) 💧", 0, 100, value=50)
        wind_speed = st.slider("Wind Speed (km/h) 🌬️", 0.0, 50.0, value=10.0)

    with col3:
        st.subheader("☀️ Solar Radiation")
        general_diffuse_flows = st.slider("General Diffuse Flows ☁️", 0.0, 500.0, value=100.0)
        diffuse_flows = st.slider("Diffuse Flows 🌤️", 0.0, 500.0, value=50.0)

    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    st.header("🔮 Predicted Power Consumption")
    try:
        update_predictions(date_input, time_input, temperature, humidity, wind_speed, general_diffuse_flows, diffuse_flows)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your inputs and try again.")

# Add energy-saving tips to sidebar
st.sidebar.header("💡 Energy Saving Tips")
st.sidebar.markdown("""
- Turn off lights when not in use 💡
- Use energy-efficient appliances ⚡
- Optimize heating and cooling 🌡️
- Unplug devices on standby 🔌
""")

# Add a footer
st.markdown("---")