import streamlit as st
import pandas as pd
import joblib

# Load the saved models and encoders
ensemble_clf = joblib.load('ensemble_clf.pkl')
encoder = joblib.load('encoder.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define custom CSS for light and dark modes
def set_page_styles(dark_mode):
    if dark_mode:
        st.markdown(
            """<style>
            body {
                background-color: #ffffff;
                color: #ffffff;
            }
            .stButton > button {
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #444444;
            }
            .stTextInput > div > input, .stSelectbox > div > div > div {
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #444444;
            }
            .stSidebar {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            .css-1aumxhk {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            </style>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<style>
            body {
                background-color: #ffffff;
                color: #000000;
            }
            .stButton > button {
                background-color: #f0f0f0;
                color: #000000;
                border: 1px solid #cccccc;
            }
            .stTextInput > div > input, .stSelectbox > div > div > div {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #cccccc;
            }
            .stSidebar {
                background-color: #f8f9fa;
                color: #000000;
            }
            .css-1aumxhk {
                background-color: #f8f9fa;
                color: #000000;
            }
            </style>""",
            unsafe_allow_html=True,
        )

# Page configuration
st.set_page_config(page_title="Laptop Recommendation System", layout="centered")

# Sidebar for theme toggle
dark_mode = st.sidebar.checkbox("Dark Mode")
set_page_styles(dark_mode)

# Streamlit app title
st.title("Laptop Recommendation System")

# Input fields for user preferences
st.header("Enter Your Preferences")
with st.form("user_preferences_form"):
    persona = st.selectbox(
        "Select Persona", ["Student", "Gamer", "Professional", "Creative", "Engineering", "Business"]
    )
    usage = st.text_input(
        "Describe Usage (e.g., Studying, Gaming, Video Editing)", "Studying, assignments, research"
    )
    processor = st.selectbox(
        "Preferred Processor", ["Intel Core i5 / AMD Ryzen 5", "Intel Core i7 / AMD Ryzen 7"]
    )
    ram = st.selectbox("Preferred RAM", ["8GB DDR4", "16GB DDR4"])
    graphics = st.selectbox(
        "Preferred Graphics", [
            "Integrated (Intel Iris Xe)",
            "NVIDIA RTX 3060 / AMD Radeon RX 6600XT",
            "NVIDIA RTX 3070 / AMD Radeon RX 6700M",
            "NVIDIA RTX 3080 / AMD Radeon RX 6800M",
            "NVIDIA RTX 3090 / AMD Radeon RX 6900M",
            "Integrated (Intel UHD / AMD Vega)",
            "Integrated (Intel Iris Xe) or NVIDIA MX550",
        ]
    )
    storage = st.selectbox(
        "Preferred Storage", [
            "256GB SSD",
            "512GB SSD",
            "1TB HDD",
            "512GB SSD + 1TB HDD",
            "1TB SSD + 1TB HDD",
            "1TB SSD + 2TB HDD",
        ]
    )
    display = st.selectbox("Preferred Display", ["13-15\" Full HD", "15-17\" QHD/4K"])
    battery = st.selectbox(
        "Battery Life Expectation", ["6-8 hours", "7-9 hours", "8-12 hours", "12+ hours"]
    )
    submit_button = st.form_submit_button(label="Get Recommendation")

# If form is submitted
if submit_button:
    # Create a DataFrame from user inputs
    new_user = pd.DataFrame({
        'Persona': [persona],
        'Usage': [usage],
        'Processor': [processor],
        'RAM': [ram],
        'Graphics': [graphics],
        'Storage': [storage],
        'Display': [display],
        'Battery Life': [battery]
    })

    # Encode the user input
    new_user_encoded = encoder.transform(new_user)

    # Predict the laptop specification label
    predicted_label = label_encoder.inverse_transform(ensemble_clf.predict(new_user_encoded))

    # Display the prediction
    st.subheader("Recommended Laptop Specification")
    st.success(f"**{predicted_label[0]}**")