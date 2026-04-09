import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model
import tensorflow as tf

# Page configuration
st.set_page_config(
    page_title="DeepFake-Detect",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 18px;
        font-weight: bold;
    }
    .real {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .fake {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='header'><h1>🔍 DeepFake-Detect</h1><p>AI-Powered Deepfake Detection</p></div>", unsafe_allow_html=True)

# Load or create model
@st.cache_resource
def load_model_cached():
    """Load the trained model (cached for performance)"""
    model_paths = [
        './model.h5',
        './best_model.h5',
        './deepfake_detector.h5',
        'model.h5',
        'best_model.h5',
        'deepfake_detector.h5'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                return load_model(path)
            except Exception as e:
                st.warning(f"Could not load model from {path}: {e}")
                continue
    
    return None

# Process image for detection
def preprocess_image(image_array, input_size=128):
    """Preprocess image for model prediction"""
    # Convert to RGB if needed
    if len(image_array.shape) == 2:  # Grayscale
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:  # RGBA
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    
    # Resize to model input size
    resized = cv2.resize(image_array, (input_size, input_size))
    
    # Normalize
    normalized = resized.astype('float32') / 255.0
    
    return np.expand_dims(normalized, axis=0)

# Main app
st.markdown("---")

# Information section
with st.expander("ℹ️ About This Tool"):
    st.write("""
    **DeepFake-Detect** uses a trained EfficientNet neural network to analyze facial images 
    and determine if they are authentic or synthetic (deepfake).
    
    The model was trained on:
    - FaceForensics++
    - Celeb-DF
    - DFDC
    - Google DFD
    - DeepFake-TIMIT
    
    Simply upload a facial image and the AI will analyze it for signs of deepfake manipulation.
    """)

st.markdown("---")

# Upload section
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a clear facial image for analysis"
    )

with col2:
    st.subheader("Example")
    if st.button("Use Sample Image"):
        # Create a sample image if uploaded_file is None
        st.session_state.use_sample = True

# Process uploaded image
if uploaded_file is not None or st.session_state.get('use_sample', False):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
    else:
        # Create a sample gradient image for demonstration
        image_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
    
    # Display image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Model loading and prediction
    st.markdown("---")
    st.subheader("🔬 Analysis Results")
    
    model = load_model_cached()
    
    if model is None:
        st.warning("""
        ⚠️ **No trained model found in the workspace.**
        
        To use the detection feature:
        1. Prepare your dataset using the pipeline scripts (00-02)
        2. Train the model: `python 03-train_cnn.py`
        3. The trained model will be saved and automatically loaded here
        
        For now, showing analysis template...
        """)
        
        # Show template result
        st.info("✅ This would show the deepfake detection score once a model is trained and available.")
        
    else:
        with st.spinner("Analyzing image..."):
            try:
                # Preprocess and predict
                processed_image = preprocess_image(image_array)
                prediction = model.predict(processed_image, verbose=0)[0][0]
                
                # Calculate confidence
                confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
                
                # Display result
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction > 0.5:
                        st.markdown(f"""
                        <div class='result-box real'>
                        ✅ AUTHENTIC<br>
                        Confidence: {confidence:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='result-box fake'>
                        ⚠️ DEEPFAKE DETECTED<br>
                        Confidence: {confidence:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Show score gauge
                    st.metric(
                        "Authenticity Score",
                        f"{prediction*100:.1f}%",
                        help="0% = Deepfake | 100% = Authentic"
                    )
                
                # Detailed analysis
                st.markdown("### Prediction Details")
                st.write(f"""
                - **Raw Score**: {prediction:.4f}
                - **Classification**: {'Authentic' if prediction > 0.5 else 'Deepfake'}
                - **Confidence**: {confidence:.2f}%
                """)
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px; margin-top: 2rem;'>
    <p>DeepFake-Detect | Open-source deepfake detection pipeline</p>
    <p><a href='https://github.com/aaronchong888/DeepFake-Detect'>GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)
