import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from PIL import Image
import io

# ============================================
# IMPORT DENGAN ERROR HANDLING YANG LEBIH BAIK
# ============================================
def safe_import():
    try:
        from utils.fall_detector import FallDetector
        return FallDetector, True
    except ImportError as e:
        st.error(f"❌ Critical Import Error: {e}")
        st.warning("""
        **Possible fixes:**
        1. Ensure `utils/` folder exists with `fall_detector.py`
        2. Check if `ultralytics` is in requirements.txt
        3. Try restarting the app
        """)
        
        # Show detailed traceback for debugging
        with st.expander("🔍 Show Detailed Error"):
            st.code(traceback.format_exc())
        
        # Create a dummy FallDetector class for demo mode
        class DummyDetector:
            def __init__(self, model_path=None, conf_threshold=0.5):
                self.model_path = model_path
                self.conf_threshold = conf_threshold
                self.total_frames = 0
                self.fall_detections = 0
                self.normal_detections = 0
            
            def detect(self, frame):
                self.total_frames += 1
                # Return dummy data for demo
                return frame, [], False
            
            def get_statistics(self):
                return {
                    'total_frames': self.total_frames,
                    'fall_detections': self.fall_detections,
                    'normal_detections': self.normal_detections
                }
        
        return DummyDetector, False

FallDetector, import_success = safe_import()

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Fall Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .sub-title {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .alert-box {
        background-color: #FFE5E5;
        border-left: 5px solid #FF4B4B;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        animation: pulse 2s infinite;
    }
    
    .normal-box {
        background-color: #E5FFE5;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .upload-area {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False  # For showing debug info


# ============================================
# SIDEBAR - DENGAN ERROR HANDLING
# ============================================
with st.sidebar:
    st.markdown("<div class='main-title'>⚙️ SETTINGS</div>", unsafe_allow_html=True)
    
    # Debug toggle
    st.session_state.debug_mode = st.checkbox("🔧 Debug Mode", value=False)
    
    st.subheader("🤖 Model Status")
    
    model_path = "best_fall_model.pt"
    model_exists = os.path.exists(model_path)
    
    if model_exists:
        file_size = os.path.getsize(model_path) / (1024*1024)
        
        col_status, col_size = st.columns(2)
        with col_status:
            st.success("✅ Found")
        with col_size:
            st.metric("Size", f"{file_size:.1f} MB")
    else:
        st.error("❌ Model not found!")
        st.info(f"Expected at: `{os.path.abspath(model_path)}`")
        
        # Option to use demo mode
        if st.button("🔄 Use Demo Mode", key="demo_mode"):
            st.session_state.detector = FallDetector(model_path="demo", conf_threshold=0.5)
            st.session_state.model_loaded = True
            st.success("Demo mode activated!")
            st.rerun()
    
    if model_exists or st.session_state.get('demo_mode', False):
        confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
        
        col_load, col_check = st.columns(2)
        
        with col_load:
            if st.button("🚀 Load Model", type="primary", use_container_width=True):
                try:
                    with st.spinner("Loading AI model..."):
                        detector = FallDetector(
                            model_path=model_path if model_exists else None,
                            conf_threshold=confidence
                        )
                        st.session_state.detector = detector
                        st.session_state.model_loaded = True
                    
                    st.success("✅ AI Model Loaded!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Load failed: {str(e)[:100]}...")
                    if st.session_state.debug_mode:
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
        
        with col_check:
            if st.button("🔍 Test Model", use_container_width=True):
                if st.session_state.detector:
                    # Create a test image
                    test_img = np.zeros((300, 400, 3), dtype=np.uint8)
                    cv2.putText(test_img, "TEST", (150, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    try:
                        result, detections, alert = st.session_state.detector.detect(test_img)
                        st.success(f"✅ Model test passed! Detections: {len(detections)}")
                    except Exception as e:
                        st.error(f"❌ Model test failed: {e}")

    
# ============================================
# MAIN CONTENT
# ============================================
st.markdown("<div class='main-title'>🚨 FALL DETECTION SYSTEM</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-powered fall detection for images and videos</div>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Detection", "📊 Results", "📋 Instructions"])

# ============================================
# TAB 1: DETECTION - DIPERBAIKI
# ============================================
with tab1:
    if not st.session_state.get('model_loaded', False):
        st.warning("⚠️ Please load the model from the sidebar first!")
        st.info("""
        **Steps:**
        1. 👈 Check if `best_fall_model.pt` exists
        2. Adjust confidence threshold if needed
        3. Click '🚀 Load Model'
        """)
        st.stop()
    
    st.subheader("Upload Media for Detection")
    
    media_type = st.radio(
        "Select media type:",
        ["📷 Image", "🎥 Video"],
        horizontal=True
    )
    
    if media_type == "📷 Image":
        # ============================================
        # IMAGE DETECTION - FIXED VERSION
        # ============================================
        st.markdown("### Image Detection")
        
        uploaded_image = st.file_uploader(
            "Upload an image (JPG, PNG, JPEG)",
            type=['jpg', 'jpeg', 'png'],
            help="Maximum file size: 200MB"
        )
        
        if uploaded_image is not None:
            try:
                # SAFETY CHECK 1: Validate upload
                if uploaded_image.size == 0:
                    st.error("❌ Uploaded file is empty!")
                    st.stop()
                
                # SAFETY CHECK 2: Try to open image
                try:
                    image = Image.open(uploaded_image)
                    image.verify()  # Verify it's a valid image
                except Exception as img_error:
                    st.error(f"❌ Invalid image file: {img_error}")
                    st.stop()
                
                # Re-open for processing (after verify)
                image = Image.open(uploaded_image)
                
                # Display columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Image**")
                    
                    # SAFETY CHECK 3: Before st.image()
                    if image is None:
                        st.error("❌ Failed to load image!")
                    else:
                        # Convert to RGB if needed
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        # PERBAIKAN 1: Ganti use_container_width
                        st.image(image, width=None, caption=f"Size: {image.size}")
                    
                    # Convert to OpenCV format SAFELY
                    try:
                        img_array = np.array(image)
                        if img_array is None or img_array.size == 0:
                            st.error("❌ Failed to convert image to array")
                            img_cv = None
                        else:
                            if len(img_array.shape) == 2:  # Grayscale
                                img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                            else:  # RGB
                                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    except Exception as conv_error:
                        st.error(f"❌ Image conversion failed: {conv_error}")
                        img_cv = None
                
                with col2:
                    st.markdown("**Detection Result**")
                    
                    # Only show detect button if we have valid image
                    if img_cv is not None:
                        if st.button("🔍 Detect Falls", use_container_width=True, key="detect_img"):
                            with st.spinner("Detecting..."):
                                try:
                                    # SAFETY CHECK 4: Before detector.detect()
                                    if st.session_state.detector is None:
                                        st.error("❌ Detector not initialized!")
                                    else:
                                        processed_frame, detections, alert_status = st.session_state.detector.detect(img_cv.copy())
                                        
                                        # SAFETY CHECK 5: Check processed frame
                                        if processed_frame is None:
                                            st.error("❌ Detection returned no result!")
                                        else:
                                            # Convert for display
                                            result_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                                            
                                            # PERBAIKAN 2: Ganti use_container_width
                                            st.image(result_rgb, width=None, caption="Detection Result")
                                            
                                            # Show results
                                            if detections:
                                                st.markdown("### 📊 Detection Details:")
                                                for i, det in enumerate(detections):
                                                    if det.get('class_name') == 'falling':
                                                        with st.container():
                                                            st.error(f"🚨 **FALL DETECTED**")
                                                            st.metric("Confidence", f"{det.get('confidence', 0):.1%}")
                                                    else:
                                                        with st.container():
                                                            st.success(f"✅ **Normal**")
                                                            st.metric("Confidence", f"{det.get('confidence', 0):.1%}")
                                            else:
                                                st.info("ℹ️ No objects detected")
                                            
                                            # Download button
                                            try:
                                                result_pil = Image.fromarray(result_rgb)
                                                buf = io.BytesIO()
                                                result_pil.save(buf, format='JPEG', quality=95)
                                                byte_im = buf.getvalue()
                                                
                                                st.download_button(
                                                    label="📥 Download Result",
                                                    data=byte_im,
                                                    file_name=f"detection_{int(time.time())}.jpg",
                                                    mime="image/jpeg",
                                                    use_container_width=True
                                                )
                                            except Exception as save_error:
                                                st.warning(f"Could not create download: {save_error}")
                                    
                                except Exception as detect_error:
                                    st.error(f"❌ Detection failed: {str(detect_error)[:200]}")
                                    if st.session_state.debug_mode:
                                        with st.expander("Detection Error Details"):
                                            st.code(traceback.format_exc())
                    else:
                        st.warning("⚠️ Cannot process - invalid image format")
                        
            except Exception as general_error:
                st.error(f"❌ Unexpected error: {str(general_error)[:150]}")
                if st.session_state.debug_mode:
                    with st.expander("Full Error Traceback"):
                        st.code(traceback.format_exc())

# ============================================
# TAB 2: RESULTS
# ============================================
with tab2:
    st.subheader("📊 Detection Results")
    
    if st.session_state.detector:
        stats = st.session_state.detector.get_statistics()
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Frames Processed", stats['total_frames'])
        col2.metric("Fall Detections", stats['fall_detections'])
        col3.metric("Normal Detections", stats['normal_detections'])
        col4.metric("Total Alerts", len(st.session_state.alert_history))
        
        # Alert History
        if st.session_state.alert_history:
            st.markdown("### 🚨 Alert History")
            
            for i, alert in enumerate(reversed(st.session_state.alert_history[-20:])):
                with st.expander(f"Alert #{len(st.session_state.alert_history)-i}", expanded=(i==0)):
                    if 'frame' in alert:
                        st.write(f"**Frame:** {alert['frame']}")
                    st.write(f"**Time:** {time.strftime('%H:%M:%S', time.localtime(alert['timestamp']))}")
                    
                    for det in alert['detections']:
                        st.write(f"- **{det['class_name'].upper()}**: {det['confidence']:.1%}")
        else:
            st.info("No alerts recorded yet. Upload and process some media to see results.")
        
        # Detailed results
        if st.session_state.processed_results:
            st.markdown("### 📋 Detailed Detection Log")
            
            with st.expander("View all detections"):
                for result in st.session_state.processed_results:
                    st.markdown(f"**Frame {result['frame']}** (t={result['time']:.2f}s)")
                    for det in result['detections']:
                        emoji = "🔴" if det['class_name'] == 'falling' else "🟢"
                        st.write(f"{emoji} {det['class_name']}: {det['confidence']:.1%}")
                    st.markdown("---")
    else:
        st.warning("Initialize detector to view results")

# ============================================
# TAB 3: INSTRUCTIONS
# ============================================
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📋 How to Use
        
        **1. Prepare Your Model**
        - Train a YOLOv8 model for fall detection
        - Export the model as `.pt` file
        - Upload it in the sidebar
        
        **2. Initialize Detector**
        - Set confidence threshold
        - Set alert threshold (for videos)
        - Click "Initialize Detector"
        
        **3. Upload Media**
        - Go to Detection tab
        - Choose Image or Video
        - Upload your file
        
        **4. Detect Falls**
        - For images: Click "Detect Falls"
        - For videos: Click "Process Video"
        - View results in real-time
        
        **5. Download Results**
        - Download processed images
        - View detection logs
        - Check alert history
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Tips for Best Results
        
        **Model Training**
        - Use diverse dataset
        - Include various lighting conditions
        - Different camera angles
        - Multiple fall types
        
        **Image Quality**
        - Good lighting
        - Clear visibility
        - Minimal motion blur
        - Proper resolution (640x640+)
        
        **Video Processing**
        - Process every 3-5 frames for speed
        - Use higher frame sampling for accuracy
        - Check preview for real-time feedback
        
        **Confidence Threshold**
        - 0.3-0.5: More detections, may have false positives
        - 0.5-0.7: Balanced (recommended)
        - 0.7-0.9: Fewer detections, high precision
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📦 Model Requirements
    
    Your model should:
    - Be trained with YOLOv8
    - Have 2 classes: 'falling' and 'normal'
    - Be exported as `.pt` format
    - Support 640x640 input size (recommended)
    
    ### 🌐 Deployment
    
    This app is optimized for **Streamlit Cloud**:
    - No webcam/camera access needed
    - Works with uploaded files only
    - Processes images and videos
    - Stores results in session
    """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>Fall Detection System</strong> • Powered by YOLOv8 & Streamlit</p>
    <p style="font-size: 0.85rem;">Upload images or videos for automatic fall detection</p>
</div>
""", unsafe_allow_html=True)