import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Concrete Crack Detection System",
    page_icon="🏗️",
    layout="centered"
)

st.title("🏗️ Concrete Crack Detection System")
st.write("Upload an image of a concrete surface, wall, or structure to check for structural cracks.")

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

uploaded_file = st.file_uploader("Choose a concrete image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Surface"):
        if model is None:
            st.error("Model file (`best_crack_model.keras`) not found or failed to load.")
        else:
            with st.spinner("Analyzing image..."):
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)[0][0]

                st.divider()
                st.subheader("Results & Analysis")

                if prediction > 0.5:
                    confidence = float(prediction) * 100
                    st.error(f"🚨 **CRACK DETECTED** (Confidence: {confidence:.2f}%)")
                    
                    if confidence > 80:
                        st.warning("**Severity:** High Severity (Immediate Attention Required)")
                        st.info("**Recommendation:** Structural assessment recommended. Apply high-strength epoxy injection.")
                    else:
                        st.warning("**Severity:** Moderate Severity")
                        st.info("**Recommendation:** Monitor crack growth and apply surface sealants.")
                else:
                    confidence = float(1 - prediction) * 100
                    st.success(f"✅ **NO CRACK DETECTED** (Confidence: {confidence:.2f}%)")
                    st.info("**Severity:** None / Surface Intact")
                    st.info("**Recommendation:** No repair needed. Regular structural maintenance recommended.")
