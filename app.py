import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from PIL import Image

# Import dengan error handling
try:
    from utils.fall_detector import FallDetector, DroidCamStream
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Fall Detection with DroidCam",
    page_icon="📱",
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
    
    .status-box {
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-connected {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    
    .status-disconnected {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
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
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'detector' not in st.session_state:
    st.session_state.detector = None

if 'run_webcam' not in st.session_state:
    st.session_state.run_webcam = False

if 'droidcam_ip' not in st.session_state:
    st.session_state.droidcam_ip = "192.168.1.5"

if 'droidcam_port' not in st.session_state:
    st.session_state.droidcam_port = 4747

if 'camera_mode' not in st.session_state:
    st.session_state.camera_mode = "droidcam"

if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []

if 'last_detection' not in st.session_state:
    st.session_state.last_detection = None

if 'camera_index' not in st.session_state:
    st.session_state.camera_index = 0

# ============================================
# HELPER FUNCTIONS
# ============================================
def test_droidcam_connection(ip, port):
    """Test DroidCam connection"""
    try:
        url = f"http://{ip}:{port}/video"
        test_cap = cv2.VideoCapture(url)
        
        if not test_cap.isOpened():
            return False, "Cannot connect to DroidCam URL", None
        
        ret, frame = test_cap.read()
        test_cap.release()
        
        if not ret or frame is None:
            return False, "Connected but cannot read frame", None
        
        return True, "Connection successful", frame
        
    except Exception as e:
        return False, f"Connection error: {str(e)}", None

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("<div class='main-title'>⚙️ SETTINGS</div>", unsafe_allow_html=True)
    
    # Model Configuration
    st.subheader("🤖 Model Configuration")
    model_path = st.text_input("Model Path", "best_fall_model.pt", 
                               help="Path to your trained YOLO model")
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05, 
                          help="Higher = more strict detection")
    
    # Alert Settings
    st.subheader("🚨 Alert Settings")
    alert_threshold = st.slider("Alert Threshold", 1, 20, 5, 
                               help="Consecutive fall frames to trigger alert")
    
    # Camera Settings
    st.subheader("📷 Camera Settings")
    camera_mode = st.radio(
        "Select Camera Source:",
        ["📱 DroidCam (WiFi)", "💻 Webcam/DroidCam USB", "📹 Video File"],
        index=0
    )
    
    if camera_mode == "📱 DroidCam (WiFi)":
        st.session_state.camera_mode = "droidcam"
        
        # DroidCam Configuration
        st.markdown("**DroidCam WiFi Configuration**")
        
        col_ip, col_port = st.columns([3, 1])
        with col_ip:
            droidcam_ip = st.text_input(
                "IP Address",
                value=st.session_state.droidcam_ip,
                help="Example: 192.168.1.5"
            )
        
        with col_port:
            droidcam_port = st.number_input(
                "Port",
                min_value=1024,
                max_value=65535,
                value=st.session_state.droidcam_port,
                help="Default: 4747"
            )
        
        st.session_state.droidcam_ip = droidcam_ip
        st.session_state.droidcam_port = droidcam_port
        
        # Test Connection Button
        if st.button("🔗 Test Connection", type="secondary", use_container_width=True):
            with st.spinner("Testing connection..."):
                success, message, frame = test_droidcam_connection(
                    st.session_state.droidcam_ip, 
                    st.session_state.droidcam_port
                )
                
                if success:
                    st.success(f"✅ {message}")
                    if frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption="DroidCam Preview", use_container_width=True)
                else:
                    st.error(f"❌ {message}")
        
        # Quick Setup Guide
        with st.expander("📱 DroidCam Setup Guide"):
            st.markdown("""
            1. Install DroidCam on phone
            2. Connect phone to same WiFi
            3. Open DroidCam app
            4. Note IP address shown
            5. Enter IP here and test
            """)
    
    elif camera_mode == "💻 Webcam/DroidCam USB":
        st.session_state.camera_mode = "webcam"
        st.info("Using camera device")
        
        # Camera Index Selection
        st.markdown("**Camera Device Index**")
        camera_index = st.number_input(
            "Camera Index",
            min_value=0,
            max_value=10,
            value=st.session_state.camera_index,
            help="Try 0 for built-in webcam, 1 or 2 for DroidCam USB"
        )
        st.session_state.camera_index = camera_index
        
        st.markdown("""
        **Common indexes:**
        - `0` = Built-in webcam
        - `1` = DroidCam USB (usually)
        - `2` = External camera
        """)
    
    else:
        st.session_state.camera_mode = "video"
        uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        if uploaded_video:
            st.session_state.uploaded_video = uploaded_video
    
    # Performance Settings
    st.subheader("⚡ Performance")
    frame_skip = st.slider("Frame Skip", 0, 5, 0, 
                          help="Skip frames for better performance (0 = no skip)")
    
    st.markdown("---")
    
    # Action Buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Initialize", type="primary", use_container_width=True):
            try:
                # Check if model exists
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

# ============================================
# MAIN CONTENT
# ============================================
st.markdown("<div class='main-title'>📱 FALL DETECTION SYSTEM</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Real-time fall detection using DroidCam and YOLOv8</div>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎥 Live Detection", "📊 Dashboard", "📋 Instructions"])

# ============================================
# TAB 1: LIVE DETECTION
# ============================================
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Live Video Feed")
        
        # Check if detector initialized
        if not st.session_state.detector:
            st.warning("⚠️ Please initialize the detector from sidebar first!")
            st.info("👈 Click '🚀 Initialize' button in the sidebar")
            st.stop()
        
        # Control buttons
        col_start, col_stop = st.columns(2)
        
        with col_start:
            if st.button("▶️ Start Detection", type="primary", use_container_width=True):
                st.session_state.run_webcam = True
        
        with col_stop:
            if st.button("⏹️ Stop Detection", type="secondary", use_container_width=True):
                st.session_state.run_webcam = False
        
        # Placeholders
        frame_placeholder = st.empty()
        alert_placeholder = st.empty()
        status_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        # ============================================
        # MAIN DETECTION LOOP
        # ============================================
        if st.session_state.run_webcam:
            # Initialize camera
            cap = None
            
            try:
                if st.session_state.camera_mode == "droidcam":
                    # DroidCam WiFi mode
                    url = f"http://{st.session_state.droidcam_ip}:{st.session_state.droidcam_port}/video"
                    cap = cv2.VideoCapture(url)
                    status_placeholder.info(f"📱 Connecting to DroidCam: {st.session_state.droidcam_ip}:{st.session_state.droidcam_port}")
                
                elif st.session_state.camera_mode == "webcam":
                    # Webcam/DroidCam USB mode
                    cap = cv2.VideoCapture(st.session_state.camera_index)
                    status_placeholder.info(f"💻 Opening camera index: {st.session_state.camera_index}")
                
                else:
                    # Video file mode
                    if 'uploaded_video' in st.session_state:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                            tmp.write(st.session_state.uploaded_video.read())
                            video_path = tmp.name
                        cap = cv2.VideoCapture(video_path)
                        status_placeholder.info("📹 Loading video file...")
                
                # Check if camera opened
                if cap is None or not cap.isOpened():
                    status_placeholder.error(f"""
                    ❌ Cannot open camera!
                    
                    **Troubleshooting:**
                    - DroidCam WiFi: Check IP and port
                    - Webcam/USB: Try different camera index (0, 1, 2)
                    - Make sure DroidCam app is running
                    - Check if another app is using the camera
                    """)
                    st.session_state.run_webcam = False
                    st.stop()
                
                status_placeholder.success("✅ Camera connected successfully!")
                
                # Frame counter for skipping
                frame_count = 0
                
                # Main loop - THIS IS THE KEY PART
                while st.session_state.run_webcam:
                    ret, frame = cap.read()
                    
                    if not ret:
                        if st.session_state.camera_mode == "video":
                            status_placeholder.info("📹 Video ended")
                            st.session_state.run_webcam = False
                            break
                        else:
                            status_placeholder.warning("⚠️ Cannot read frame, retrying...")
                            time.sleep(0.5)
                            continue
                    
                    # Skip frames if needed
                    frame_count += 1
                    if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                        continue
                    
                    # Flip frame for webcam (mirror effect)
                    if st.session_state.camera_mode == "webcam":
                        frame = cv2.flip(frame, 1)
                    
                    # Detect falls
                    try:
                        processed_frame, detections, alert_status = st.session_state.detector.detect(frame)
                        
                        # Store detection
                        st.session_state.last_detection = {
                            'time': time.strftime("%H:%M:%S"),
                            'detections': detections,
                            'alert': alert_status,
                            'frame': processed_frame.copy()
                        }
                        
                        # Record alert
                        if alert_status:
                            st.session_state.alert_history.append({
                                'timestamp': time.time(),
                                'detections': detections
                            })
                        
                        # Display frame
                        display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                        
                        # Display alert
                        if alert_status:
                            alert_duration = time.time() - st.session_state.detector.alert_start_time
                            alert_placeholder.markdown(f"""
                            <div class="alert-box">
                                <h2>🚨 FALL DETECTED!</h2>
                                <p style="font-size: 1.2rem;">Active for {alert_duration:.1f} seconds</p>
                                <p><strong>⚠️ IMMEDIATE ATTENTION REQUIRED!</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            if detections:
                                det_text = " | ".join([
                                    f"**{d['class_name'].upper()}** ({d['confidence']:.0%})"
                                    for d in detections
                                ])
                                alert_placeholder.markdown(f"""
                                <div class="normal-box">
                                    <h3>✅ Monitoring Normal</h3>
                                    <p>{det_text}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                alert_placeholder.markdown("""
                                <div class="normal-box">
                                    <h3>👁️ No Person Detected</h3>
                                    <p>Make sure person is visible in frame</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Show quick stats
                        stats = st.session_state.detector.get_statistics()
                        stats_placeholder.text(f"FPS: {stats['fps']:.1f} | Frames: {stats['total_frames']} | Falls: {stats['fall_detections']}")
                        
                    except Exception as e:
                        alert_placeholder.error(f"Detection error: {str(e)}")
                        break
                    
                    # Small delay to prevent overwhelming
                    time.sleep(0.01)
                
                # Cleanup
                if cap is not None:
                    cap.release()
                
                if st.session_state.camera_mode == "video" and 'video_path' in locals():
                    try:
                        os.unlink(video_path)
                    except:
                        pass
                
                status_placeholder.info("⏹️ Detection stopped")
            
            except Exception as e:
                status_placeholder.error(f"❌ Error: {str(e)}")
                st.session_state.run_webcam = False
                if cap is not None:
                    cap.release()
        
        else:
            # Not running
            frame_placeholder.info("👆 Click '▶️ Start Detection' to begin monitoring")
    
    with col2:
        st.subheader("📈 Statistics")
        
        if st.session_state.detector:
            # Create a placeholder for live stats
            stats_container = st.container()
            
            with stats_container:
                stats = st.session_state.detector.get_statistics()
                
                # FPS
                st.markdown(f"""
                <div class="metric-card">
                    <h3>FPS</h3>
                    <div class="value">{stats['fps']:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Fall Rate
                fall_rate = stats['fall_ratio'] * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Fall Rate</h3>
                    <div class="value">{fall_rate:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Other metrics
                st.metric("Total Frames", stats['total_frames'])
                st.metric("Fall Detections", stats['fall_detections'])
                st.metric("Normal", stats['normal_detections'])
                
                # Alert status
                if stats['alert_active']:
                    st.error(f"🚨 Alert Active ({stats['alert_duration']:.1f}s)")
                else:
                    st.success("✅ No Active Alerts")
                
                # Last detection
                if st.session_state.last_detection:
                    with st.expander("Last Detection"):
                        st.write(f"Time: {st.session_state.last_detection['time']}")
                        for det in st.session_state.last_detection['detections']:
                            emoji = "🔴" if det['class_name'] == 'falling' else "🟢"
                            st.write(f"{emoji} {det['class_name']}: {det['confidence']:.0%}")

# ============================================
# TAB 2: DASHBOARD
# ============================================
with tab2:
    st.subheader("📊 Dashboard")
    
    if st.session_state.detector:
        stats = st.session_state.detector.get_statistics()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            uptime = stats['total_frames'] / max(stats['fps'], 1)
            st.metric("Uptime", f"{uptime:.1f}s")
        
        with col2:
            accuracy = ((stats['fall_detections'] + stats['normal_detections']) / 
                       max(stats['total_frames'], 1) * 100)
            st.metric("Detection Rate", f"{accuracy:.1f}%")
        
        with col3:
            st.metric("Total Alerts", len(st.session_state.alert_history))
        
        with col4:
            st.metric("Avg FPS", f"{stats['fps']:.1f}")
        
        # Alert History
        st.subheader("🚨 Alert History")
        if st.session_state.alert_history:
            for i, alert in enumerate(reversed(st.session_state.alert_history[-10:])):
                alert_time = time.strftime("%H:%M:%S", 
                                          time.localtime(alert['timestamp']))
                time_ago = time.time() - alert['timestamp']
                
                with st.expander(f"🚨 Alert #{len(st.session_state.alert_history)-i} at {alert_time} ({time_ago:.0f}s ago)", expanded=(i==0)):
                    for det in alert['detections']:
                        st.write(f"- **{det['class_name'].upper()}**: {det['confidence']:.1%}")
        else:
            st.info("No alerts recorded yet")
    else:
        st.warning("Initialize detector to see dashboard")

# ============================================
# TAB 3: INSTRUCTIONS
# ============================================
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Setup Guide")
        st.markdown("""
        ### 🚀 Quick Start:
        
        **Option 1: DroidCam WiFi**
        1. Install DroidCam app on phone
        2. Connect phone & PC to same WiFi
        3. Open DroidCam, note IP address
        4. Enter IP in sidebar, test connection
        5. Initialize detector
        6. Start detection
        
        **Option 2: DroidCam USB**
        1. Install DroidCam client on PC
        2. Connect phone via USB
        3. Select "Webcam/DroidCam USB" mode
        4. Try camera index 1 or 2
        5. Initialize detector
        6. Start detection
        
        **Option 3: Built-in Webcam**
        1. Select "Webcam/DroidCam USB"
        2. Use camera index 0
        3. Initialize detector
        4. Start detection
        """)
    
    with col2:
        st.subheader("🔧 Troubleshooting")
        st.markdown("""
        ### Common Issues:
        
        **"Cannot open camera"**
        - ✅ For WiFi: Check IP/port, same network
        - ✅ For USB: Try index 0, 1, or 2
        - ✅ Close other apps using camera
        - ✅ Restart DroidCam app
        
        **"Running but no video"**
        - ✅ Check if detector initialized
        - ✅ Click Start button
        - ✅ Wait a few seconds
        - ✅ Check browser console for errors
        
        **"Slow/Laggy video"**
        - ✅ Increase Frame Skip (sidebar)
        - ✅ Use lower resolution in DroidCam
        - ✅ Better WiFi signal
        - ✅ Close other applications
        
        **"No detections"**
        - ✅ Ensure person clearly visible
        - ✅ Lower confidence threshold
        - ✅ Better lighting
        - ✅ Check model file exists
        """)
    
    st.subheader("📱 Camera Index Guide")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Index 0**
        - Built-in webcam
        - Default camera
        """)
    
    with col2:
        st.info("""
        **Index 1**
        - DroidCam USB (common)
        - External camera
        """)
    
    with col3:
        st.info("""
        **Index 2**
        - DroidCam USB (alt)
        - Second external camera
        """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Fall Detection System v2.0 • Powered by YOLOv8 & Streamlit</p>
    <p style="font-size: 0.8rem;">Press Stop button to end detection • Check sidebar for settings</p>
</div>
""", unsafe_allow_html=True)