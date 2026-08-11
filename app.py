import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page setup for wide dashboard view
st.set_page_config(
    page_title="Concrete Crack Detection System",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Concrete Crack Detection System")
st.write("Upload a concrete surface image to detect structural cracks and get maintenance recommendations.")

MODEL_PATH = "best_crack_model.keras"

@st.cache_resource
def load_crack_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            st.error(f"Model load error: {e}")
            return None
    return None

model = load_crack_model()

# Split page into 2 equal columns (Gradio layout style)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📷 Input Image")
    uploaded_file = st.file_uploader("Upload Concrete / Wall Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Surface", use_column_width=True)
        analyze_btn = st.button("Submit / Analyze", type="primary", use_container_width=True)
    else:
        analyze_btn = False

with col2:
    st.subheader("📊 Detection & Severity Results")
    
    if uploaded_file is not None and analyze_btn:
        if model is None:
            st.error("Model file (`best_crack_model.keras`) not found or failed to load.")
        else:
            with st.spinner("Processing image through AI model..."):
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)[0][0]

                if prediction > 0.5:
                    confidence = float(prediction) * 100
                    st.error(f"🚨 **STATUS: CRACK DETECTED**\n\n**Confidence:** {confidence:.2f}%")
                    
                    if confidence > 80:
                        st.warning("**Severity:** High Severity (Immediate Attention Required)")
                        st.info("**Recommendation:** Structural assessment recommended. Apply high-strength epoxy injection.")
                    else:
                        st.warning("**Severity:** Moderate Severity")
                        st.info("**Recommendation:** Monitor crack growth and apply surface sealants.")
                else:
                    confidence = float(1 - prediction) * 100
                    st.success(f"✅ **STATUS: NO CRACK DETECTED**\n\n**Confidence:** {confidence:.2f}%")
                    st.info("**Severity:** None / Surface Intact")
                    st.info("**Recommendation:** No repair needed. Regular structural maintenance recommended.")
    else:
        st.info("👈 Upload an image on the left panel and click 'Submit' to view detection results here.")
