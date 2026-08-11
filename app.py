import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Set dark page layout
st.set_page_config(
    page_title="AI Crack Detection System",
    page_icon="🏗️",
    layout="wide"
)

# Custom Gradio Dark Theme Styling
st.markdown("""
<style>
    /* Dark Background Override */
    .stApp {
        background-color: #0b0f19;
        color: #ffffff;
    }
    
    /* Box Container styling like Gradio */
    div[data-testid="column"] {
        background-color: #121826;
        border: 1px solid #1f293d;
        border-radius: 8px;
        padding: 18px;
    }

    /* Buttons Styling */
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        height: 45px;
    }
    
    /* Primary Submit Button */
    div[data-testid="column"]:nth-child(1) .stButton>button[kind="primary"] {
        background-color: #ff5500;
        color: white;
        border: none;
    }
    div[data-testid="column"]:nth-child(1) .stButton>button[kind="primary"]:hover {
        background-color: #e04b00;
    }

    /* Hide Streamlit Header & Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Title & Description
st.markdown("## 🏗️ AI Crack Detection System")
st.markdown("<p style='color: #9ca3af;'>MobileNetV2-based concrete crack detection tool.</p>", unsafe_allow_html=True)

MODEL_PATH = "best_crack_model.keras"

@st.cache_resource
def load_crack_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            return None
    return None

model = load_crack_model()

# Gradio Side-by-Side 2 Column Layout
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown("<h5 style='color: #9ca3af;'>🖼️ Upload Concrete/Wall Image</h5>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)
    
    b_col1, b_col2 = st.columns([1, 1])
    with b_col1:
        clear_btn = st.button("Clear", use_container_width=True)
    with b_col2:
        submit_btn = st.button("Submit", type="primary", use_container_width=True)

with col2:
    st.markdown("<h5 style='color: #9ca3af;'>📊 Detection Result</h5>", unsafe_allow_html=True)
    
    if uploaded_file is not None and submit_btn:
        if model is None:
            st.error("Model file (`best_crack_model.keras`) load nahi ho pa rahi hai.")
        else:
            with st.spinner("Analyzing image..."):
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)[0][0]

                st.markdown("---")
                if prediction > 0.5:
                    confidence = float(prediction) * 100
                    st.error(f"### 🚨 Status: CRACK DETECTED\n**Confidence:** {confidence:.2f}%")
                    
                    st.markdown("<h5 style='color: #9ca3af; margin-top:20px;'>Severity Analysis</h5>", unsafe_allow_html=True)
                    if confidence > 80:
                        st.warning("High Severity (Immediate Attention Required)")
                        st.markdown("<h5 style='color: #9ca3af; margin-top:15px;'>Maintenance Recommendation</h5>", unsafe_allow_html=True)
                        st.info("Structural assessment recommended. Apply high-strength epoxy injection.")
                    else:
                        st.warning("Moderate Severity")
                        st.markdown("<h5 style='color: #9ca3af; margin-top:15px;'>Maintenance Recommendation</h5>", unsafe_allow_html=True)
                        st.info("Monitor crack growth and apply surface sealants.")
                else:
                    confidence = float(1 - prediction) * 100
                    st.success(f"### ✅ Status: NO CRACK DETECTED\n**Confidence:** {confidence:.2f}%")
                    
                    st.markdown("<h5 style='color: #9ca3af; margin-top:20px;'>Severity Analysis</h5>", unsafe_allow_html=True)
                    st.info("None / Surface Intact")
                    
                    st.markdown("<h5 style='color: #9ca3af; margin-top:15px;'>Maintenance Recommendation</h5>", unsafe_allow_html=True)
                    st.info("No repair needed. Regular structural maintenance recommended.")
    else:
        st.markdown("<div style='height: 250px; display: flex; align-items: center; justify-content: center; border: 1px dashed #2e384d; border-radius: 6px; color: #6b7280;'>Results will appear here after clicking Submit</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("🔗 Share via Link", disabled=True, use_container_width=True)
