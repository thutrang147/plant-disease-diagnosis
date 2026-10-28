"""Minimal Streamlit interface for the plant disease diagnosis project."""

import streamlit as st


st.set_page_config(page_title="Plant Disease Diagnosis", page_icon="🌿")
st.title("Plant Disease Diagnosis")
st.write(
    "Upload a plant image to prepare it for diagnosis. "
    "Connect a trained model in this app to enable predictions."
)
st.file_uploader("Choose a plant image", type=("jpg", "jpeg", "png"))
