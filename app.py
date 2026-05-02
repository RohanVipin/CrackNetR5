import streamlit as st
from fastai.vision.all import *
import pathlib
import platform

# Fix for Windows path issue in fastai
plt = platform.system()
if plt == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

# Load models
binary_model = load_learner('pothole_model.pkl')
severity_model = load_learner('Severity_classifier.pkl')

# Severity mapping
severity_map = {
    'S': 'Severe Pothole',
    'A': 'High Severity',
    'B': 'Moderate Pothole',
    'C': 'Minor Pothole'
}

st.title("CrackNet-R5: Road Damage Detection System")
st.write("Upload an image of a pothole and the model will predict severity level.")

# Upload image
uploaded_file = st.file_uploader("Upload Pothole Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    img = PILImage.create(uploaded_file)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Step 1: Binary Classification
    pred_bin, pred_idx_bin, probs_bin = binary_model.predict(img)

    st.subheader("Pothole Detection Result")
    st.write(f"Pothole Present: **{pred_bin}**")

    # Step 2: Severity only if pothole exists
    if str(pred_bin) == '1' or str(pred_bin).lower() == 'pothole':
        pred_sev, pred_idx_sev, probs_sev = severity_model.predict(img)

        severity_text = severity_map.get(str(pred_sev), "Unknown")
        confidence = probs_sev[pred_idx_sev].item()

        st.subheader("Severity Result")
        st.write(f"Severity Level: **{severity_text}**")
        st.write(f"Confidence: **{confidence:.2f}**")
    else:
        st.subheader("Severity Result")
        st.write("Severity Level: **None (No Pothole Detected)**")