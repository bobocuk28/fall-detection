# app.py - DEPLOY READY VERSION
import streamlit as st
import sys
import os
import tempfile
import time
from PIL import Image

# ============================================
# IMPORT DENGAN ERROR HANDLING UNTUK CLOUD
# ============================================
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError as e:
    CV2_AVAILABLE = False
    import numpy as np
    st.warning(f"⚠️ OpenCV not available: {e}. Some features disabled.")

# ============================================
# PAGE CONFIGURATION (HARUS DI AWAL)
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
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DETECT CLOUD ENVIRONMENT
# ============================================
IS_CLOUD = os.path.exists('/mount') or 'STREAMLIT_SHARING_MODE' in os.environ

# ============================================
# IMPORT FALL DETECTOR DENGAN ERROR HANDLING
# ============================================
FallDetector = None
DroidCamStream = None

if CV2_AVAILABLE:
    try:
        from utils.fall_detector import FallDetector, DroidCamStream
    except ImportError as e:
        st.warning(f"⚠️ Fall detector module not available: {e}")
else:
    st.info("ℹ️ Running in limited mode - OpenCV required for full functionality")

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'detector' not in st.session_state:
    st.session_state.detector = None

if 'run_webcam' not in st.session_state:
    st.session_state.run_webcam = False

if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []

if 'last_detection' not in st.session_state:
    st.session_state.last_detection = None

# ============================================
# MAIN TITLE
# ============================================
st.markdown("<div class='main-title'>🚨 AI FALL DETECTION SYSTEM</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>University Final Project | Powered by YOLOv8</div>", unsafe_allow_html=True)

if IS_CLOUD:
    st.success("🌐 **CLOUD MODE ACTIVE** - File Upload Available")
    st.info("""
    **Features Available in Cloud:**
    - 📹 Upload video files (MP4, AVI, MOV)
    - 📸 Upload images for testing
    - 📊 View analytics and results
    - 🎯 Test with sample data
    
    **Note:** Live camera features require local installation.
    """)

# ============================================
# SIDEBAR - CLOUD OPTIMIZED
# ============================================
with st.sidebar:
    st.markdown("### ⚙️ SETTINGS")
    
    # Model Configuration
    st.subheader("🤖 Model Configuration")
    
    # Default model paths
    default_model = "best_fall_model.onnx" if os.path.exists("best_fall_model.onnx") else "best_fall_model.pt"
    
    model_path = st.text_input("Model Path", default_model, 
                               help="Path to your trained model")
    
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
    
    # Alert Settings
    st.subheader("🚨 Alert Settings")
    alert_threshold = st.slider("Alert Threshold", 1, 20, 5)
    
    # Camera/File Settings
    st.subheader("📷 Input Source")
    
    if IS_CLOUD:
        input_mode = st.radio("Select Input:", ["📹 Upload Video File", "📸 Upload Image"])
    else:
        input_mode = st.radio("Select Input:", 
                            ["📹 Upload Video File", "📸 Upload Image", "📱 DroidCam", "💻 Webcam"])
    
    # Performance
    st.subheader("⚡ Performance")
    frame_skip = st.slider("Frame Skip", 0, 5, 0)
    
    st.markdown("---")
    
    # Action Buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Initialize", type="primary", use_container_width=True):
            if not CV2_AVAILABLE:
                st.error("❌ OpenCV not available. Please check requirements.")
            elif FallDetector is None:
                st.error("❌ Fall detector module not found.")
            else:
                try:
                    if not os.path.exists(model_path):
                        st.error(f"❌ Model not found: {model_path}")
                    else:
                        st.session_state.detector = FallDetector(
                            model_path=model_path,
                            conf_threshold=confidence
                        )
                        st.session_state.detector.alert_threshold = alert_threshold
                        st.success("✅ Detector initialized!")
                except Exception as e:
                    st.error(f"❌ Initialization failed: {str(e)}")
    
    with col2:
        if st.button("🔄 Reset", type="secondary", use_container_width=True):
            if st.session_state.detector:
                st.session_state.detector.reset_statistics()
            st.session_state.alert_history = []
            st.session_state.last_detection = None
            st.session_state.run_webcam = False
            st.success("✅ Reset complete!")
    
    # Cloud info
    if IS_CLOUD:
        st.markdown("---")
        st.info("""
        **Cloud Limitations:**
        - Max file size: 200MB
        - Processing time: ~30s per minute of video
        - Live camera: Not available
        """)

# ============================================
# MAIN CONTENT AREA
# ============================================
tab1, tab2, tab3 = st.tabs(["🎯 Detection", "📊 Analytics", "📖 Guide"])

# TAB 1: DETECTION
with tab1:
    st.subheader("Detection Interface")
    
    if not CV2_AVAILABLE:
        st.error("""
        ⚠️ **OpenCV not available!**
        
        For cloud deployment, please ensure:
        1. `opencv-python-headless` is in requirements.txt
        2. Model file exists in repository
        3. All dependencies are installed
        
        **Current mode:** Limited functionality
        """)
    
    # File upload section
    uploaded_file = None
    
    if "Upload Video" in input_mode:
        uploaded_file = st.file_uploader("Upload Video File", 
                                       type=['mp4', 'avi', 'mov'],
                                       help="Max 200MB")
    elif "Upload Image" in input_mode:
        uploaded_file = st.file_uploader("Upload Image", 
                                       type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        # Save to temp file
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name
        
        # Display file
        if uploaded_file.type.startswith('video'):
            st.video(file_path)
            
            if st.button("🎬 Analyze Video", type="primary"):
                if not st.session_state.detector:
                    st.warning("Please initialize detector first!")
                elif not CV2_AVAILABLE:
                    st.error("OpenCV not available for video processing")
                else:
                    with st.spinner("Analyzing video..."):
                        # Initialize video capture
                        cap = cv2.VideoCapture(file_path)
                        
                        if not cap.isOpened():
                            st.error("Cannot open video file")
                        else:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            results = []
                            
                            # Get video info
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            
                            # Process video
                            frame_count = 0
                            alert_count = 0
                            
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                # Skip frames if needed
                                if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                                    frame_count += 1
                                    continue
                                
                                # Process frame
                                processed, detections, alert = st.session_state.detector.detect(frame)
                                
                                if alert:
                                    alert_count += 1
                                
                                # Update progress
                                progress = (frame_count + 1) / total_frames
                                progress_bar.progress(progress)
                                status_text.text(f"Processing frame {frame_count + 1}/{total_frames}")
                                
                                frame_count += 1
                                if frame_count >= 100:  # Limit for demo
                                    break
                            
                            cap.release()
                            
                            # Show results
                            st.success(f"✅ Analysis complete! Processed {frame_count} frames")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Frames Analyzed", frame_count)
                            with col2:
                                st.metric("Falls Detected", alert_count)
                            with col3:
                                st.metric("Processing Time", f"{frame_count/fps:.1f}s")
                            
                            if alert_count > 0:
                                st.error(f"🚨 **ALERT:** {alert_count} potential falls detected!")
                            else:
                                st.success("✅ No falls detected")
        
        else:  # Image file
            img = Image.open(file_path)
            st.image(img, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Detect Falls", type="primary"):
                if not st.session_state.detector:
                    st.warning("Please initialize detector first!")
                elif not CV2_AVAILABLE:
                    st.error("OpenCV not available for image processing")
                else:
                    with st.spinner("Detecting..."):
                        # Convert PIL to OpenCV
                        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        
                        # Detect
                        processed_img, detections, alert = st.session_state.detector.detect(img_cv)
                        
                        # Convert back to RGB for display
                        result_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                        
                        # Show results
                        st.subheader("Detection Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(result_rgb, caption="Processed Image", use_container_width=True)
                        
                        with col2:
                            if detections:
                                st.success(f"✅ {len(detections)} object(s) detected")
                                for det in detections:
                                    status = "🔴 FALLING" if det['class_name'] == 'falling' else "🟢 NORMAL"
                                    st.write(f"**{status}**: {det['confidence']:.1%} confidence")
                                
                                if alert:
                                    st.error("🚨 **FALL DETECTED!** Immediate attention required!")
                                else:
                                    st.success("✅ Situation normal - No falls detected")
                            else:
                                st.warning("⚠️ No objects detected")
        
        # Cleanup
        os.unlink(file_path)
    
    elif not IS_CLOUD and "DroidCam" in input_mode:
        # DroidCam interface (local only)
        st.info("📱 DroidCam Mode - Requires local installation")
        
        col_ip, col_port = st.columns(2)
        with col_ip:
            droidcam_ip = st.text_input("IP Address", "192.168.1.5")
        with col_port:
            droidcam_port = st.number_input("Port", 4747)
        
        if st.button("Connect DroidCam"):
            if not CV2_AVAILABLE:
                st.error("OpenCV not available")
            else:
                try:
                    url = f"http://{droidcam_ip}:{droidcam_port}/video"
                    cap = cv2.VideoCapture(url)
                    
                    if cap.isOpened():
                        st.success("✅ DroidCam connected!")
                        ret, frame = cap.read()
                        if ret:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, caption="DroidCam Preview")
                        cap.release()
                    else:
                        st.error("❌ Cannot connect to DroidCam")
                except Exception as e:
                    st.error(f"Connection error: {e}")
    
    elif not IS_CLOUD and "Webcam" in input_mode:
        # Webcam interface (local only)
        st.info("💻 Webcam Mode - Requires local installation")
        
        if st.button("Start Webcam"):
            if not CV2_AVAILABLE:
                st.error("OpenCV not available")
            else:
                st.warning("Webcam feature requires local execution")

# TAB 2: ANALYTICS
with tab2:
    st.subheader("Analytics Dashboard")
    
    if st.session_state.detector:
        stats = st.session_state.detector.get_statistics()
        
        # Metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Frames</h3>
                <div class="value">{}</div>
            </div>
            """.format(stats['total_frames']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Falls Detected</h3>
                <div class="value">{}</div>
            </div>
            """.format(stats['fall_detections']), unsafe_allow_html=True)
        
        with col3:
            fall_rate = stats['fall_ratio'] * 100
            st.markdown("""
            <div class="metric-card">
                <h3>Fall Rate</h3>
                <div class="value">{:.1f}%</div>
            </div>
            """.format(fall_rate), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>FPS</h3>
                <div class="value">{:.1f}</div>
            </div>
            """.format(stats['fps']), unsafe_allow_html=True)
        
        # Alert history
        st.subheader("🚨 Alert History")
        if st.session_state.alert_history:
            for i, alert in enumerate(reversed(st.session_state.alert_history[-5:])):
                with st.expander(f"Alert #{len(st.session_state.alert_history)-i}", expanded=(i==0)):
                    st.write(f"Time: {time.ctime(alert['timestamp'])}")
                    for det in alert['detections']:
                        st.write(f"- {det['class_name'].upper()}: {det['confidence']:.1%}")
        else:
            st.info("No alerts recorded yet")
    else:
        st.warning("Initialize detector to see analytics")

# TAB 3: GUIDE
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Quick Start")
        
        if IS_CLOUD:
            st.markdown("""
            **For Cloud Usage:**
            1. Initialize detector from sidebar
            2. Upload video or image file
            3. Click "Analyze" button
            4. View results and analytics
            
            **Supported Formats:**
            - Videos: MP4, AVI, MOV
            - Images: JPG, PNG
            - Max size: 200MB
            """)
        else:
            st.markdown("""
            **For Local Usage:**
            1. Install dependencies
            2. Initialize detector
            3. Select input source
            4. Start detection
            
            **Input Options:**
            - DroidCam (WiFi/USB)
            - Webcam
            - Video files
            - Image files
            """)
    
    with col2:
        st.subheader("🔧 Troubleshooting")
        st.markdown("""
        **Common Issues:**
        
        **"Module not found"**
        - Check requirements.txt
        - Use `opencv-python-headless` for cloud
        
        **"Model not found"**
        - Ensure model file is in repository
        - Use .onnx format for smaller size
        
        **"Slow processing"**
        - Increase Frame Skip
        - Use smaller video files
        - Reduce confidence threshold
        """)
    
    st.subheader("📱 About This Project")
    st.markdown("""
    **Fall Detection AI System**
    
    - **Framework:** YOLOv8 Object Detection
    - **Interface:** Streamlit Web Application
    - **Deployment:** Streamlit Cloud / Hugging Face
    - **Purpose:** University Final Project
    
    **Features:**
    - Real-time fall detection
    - Multiple input sources
    - Analytics dashboard
    - Alert system
    """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🎓 Fall Detection System • University Project</p>
    <p style="font-size: 0.9rem;">Deployed on Streamlit Cloud • Powered by YOLOv8</p>
</div>
""", unsafe_allow_html=True)