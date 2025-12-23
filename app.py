import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from PIL import Image
import io

# Import dengan error handling
try:
    from utils.fall_detector import FallDetector
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.info("Make sure 'utils/fall_detector.py' exists in your repository")
    st.stop()

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

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("<div class='main-title'>⚙️ SETTINGS</div>", unsafe_allow_html=True)
    
    # Model Configuration
    st.subheader("🤖 Model Configuration")
    
    # Model file uploader
    model_file = st.file_uploader(
        "Upload Model (.pt file)", 
        type=['pt'],
        help="Upload your trained YOLOv8 model (.pt file)"
    )
    
    if model_file is not None:
        # Save uploaded model temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
            tmp_model.write(model_file.read())
            model_path = tmp_model.name
    else:
        # Default model path (if model is in repo)
        model_path = "best_fall_model.pt"
        if not os.path.exists(model_path):
            st.warning("⚠️ No model found. Please upload a model file above.")
            model_path = None
    
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05, 
                          help="Higher = more strict detection")
    
    # Alert Settings
    st.subheader("🚨 Alert Settings")
    alert_threshold = st.slider("Alert Threshold (for video)", 1, 20, 5, 
                               help="Consecutive fall frames to trigger alert")
    
    st.markdown("---")
    
    # Initialize Button
    if st.button("🚀 Initialize Detector", type="primary", use_container_width=True):
        if model_path and os.path.exists(model_path):
            try:
                with st.spinner("Loading model..."):
                    st.session_state.detector = FallDetector(
                        model_path=model_path,
                        conf_threshold=confidence
                    )
                    st.session_state.detector.alert_threshold = alert_threshold
                st.success("✅ Detector initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize: {str(e)}")
        else:
            st.error("❌ Please upload a model file first")
    
    # Reset Button
    if st.button("🔄 Reset All", type="secondary", use_container_width=True):
        if st.session_state.detector:
            st.session_state.detector.reset_statistics()
        st.session_state.alert_history = []
        st.session_state.processed_results = []
        st.success("✅ Reset complete!")
        st.rerun()
    
    # Info
    st.markdown("---")
    st.info("""
    **How to use:**
    1. Upload your model (.pt)
    2. Click Initialize
    3. Go to Detection tab
    4. Upload image/video
    5. View results!
    """)

# ============================================
# MAIN CONTENT
# ============================================
st.markdown("<div class='main-title'>🚨 FALL DETECTION SYSTEM</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-powered fall detection for images and videos</div>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Detection", "📊 Results", "📋 Instructions"])

# ============================================
# TAB 1: DETECTION
# ============================================
with tab1:
    if not st.session_state.detector:
        st.warning("⚠️ Please initialize the detector from the sidebar first!")
        st.info("👈 Upload your model and click 'Initialize Detector'")
        st.stop()
    
    st.subheader("Upload Media for Detection")
    
    # Media type selector
    media_type = st.radio(
        "Select media type:",
        ["📷 Image", "🎥 Video"],
        horizontal=True
    )
    
    if media_type == "📷 Image":
        # ============================================
        # IMAGE DETECTION
        # ============================================
        st.markdown("### Image Detection")
        
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to detect falls"
        )
        
        if uploaded_image is not None:
            # Display columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                # Read and display original image
                image = Image.open(uploaded_image)
                st.image(image, use_container_width=True)
                
                # Convert to OpenCV format
                img_array = np.array(image)
                if len(img_array.shape) == 2:  # Grayscale
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                else:  # RGB
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            with col2:
                st.markdown("**Detection Result**")
                
                if st.button("🔍 Detect Falls", use_container_width=True):
                    with st.spinner("Detecting..."):
                        # Detect
                        processed_frame, detections, alert_status = st.session_state.detector.detect(img_cv.copy())
                        
                        # Convert back to RGB for display
                        result_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display result
                        st.image(result_rgb, use_container_width=True)
                        
                        # Show detection info
                        if detections:
                            st.markdown("### Detection Details:")
                            for i, det in enumerate(detections):
                                if det['class_name'] == 'falling':
                                    st.error(f"🚨 **FALL DETECTED** - Confidence: {det['confidence']:.1%}")
                                else:
                                    st.success(f"✅ **Normal** - Confidence: {det['confidence']:.1%}")
                        else:
                            st.info("ℹ️ No person detected in the image")
                        
                        # Download button
                        result_pil = Image.fromarray(result_rgb)
                        buf = io.BytesIO()
                        result_pil.save(buf, format='JPEG')
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="📥 Download Result",
                            data=byte_im,
                            file_name=f"detection_result_{int(time.time())}.jpg",
                            mime="image/jpeg"
                        )
    
    else:
        # ============================================
        # VIDEO DETECTION
        # ============================================
        st.markdown("### Video Detection")
        
        uploaded_video = st.file_uploader(
            "Upload a video",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video to detect falls"
        )
        
        if uploaded_video is not None:
            # Save video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                tmp_video.write(uploaded_video.read())
                video_path = tmp_video.name
            
            st.success("✅ Video uploaded successfully!")
            
            # Video info
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            col_info1, col_info2, col_info3 = st.columns(3)
            col_info1.metric("Total Frames", total_frames)
            col_info2.metric("FPS", fps)
            col_info3.metric("Duration", f"{duration:.1f}s")
            
            # Processing options
            st.markdown("### Processing Options")
            col_opt1, col_opt2 = st.columns(2)
            
            with col_opt1:
                process_every_n = st.slider(
                    "Process every N frames",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Process every Nth frame for faster processing"
                )
            
            with col_opt2:
                show_preview = st.checkbox("Show live preview", value=True)
            
            # Process button
            if st.button("🎬 Process Video", use_container_width=True, type="primary"):
                # Placeholders
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                if show_preview:
                    preview_placeholder = st.empty()
                
                alert_placeholder = st.empty()
                
                # Open video
                cap = cv2.VideoCapture(video_path)
                
                frame_count = 0
                processed_count = 0
                fall_frames = []
                all_detections = []
                
                # Process video
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Process every N frames
                    if frame_count % process_every_n == 0:
                        # Detect
                        processed_frame, detections, alert_status = st.session_state.detector.detect(frame.copy())
                        
                        # Store results
                        if alert_status:
                            fall_frames.append(frame_count)
                        
                        if detections:
                            all_detections.append({
                                'frame': frame_count,
                                'time': frame_count / fps,
                                'detections': detections,
                                'alert': alert_status
                            })
                        
                        # Update preview
                        if show_preview and processed_count % 5 == 0:
                            preview_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            preview_placeholder.image(preview_rgb, caption=f"Frame {frame_count}/{total_frames}", use_container_width=True)
                        
                        processed_count += 1
                    
                    # Update progress
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {frame_count}/{total_frames} frames ({progress*100:.1f}%)")
                
                cap.release()
                
                # Show results
                st.success(f"✅ Processing complete! Processed {processed_count} frames")
                
                # Summary
                st.markdown("### Detection Summary")
                col_res1, col_res2, col_res3 = st.columns(3)
                
                col_res1.metric("Total Detections", len(all_detections))
                col_res2.metric("Fall Alerts", len(fall_frames))
                col_res3.metric("Alert Rate", f"{len(fall_frames)/max(processed_count,1)*100:.1f}%")
                
                # Show fall alerts
                if fall_frames:
                    alert_placeholder.error(f"🚨 **FALL DETECTED** in {len(fall_frames)} frames!")
                    
                    st.markdown("### Fall Detection Timeline")
                    for i, frame_num in enumerate(fall_frames[:10]):  # Show first 10
                        time_stamp = frame_num / fps
                        st.warning(f"⚠️ Fall #{i+1} at frame {frame_num} ({time_stamp:.2f}s)")
                    
                    if len(fall_frames) > 10:
                        st.info(f"... and {len(fall_frames)-10} more fall detections")
                else:
                    alert_placeholder.success("✅ No falls detected in the video")
                
                # Store results
                st.session_state.processed_results = all_detections
                st.session_state.alert_history.extend([{
                    'timestamp': time.time(),
                    'frame': d['frame'],
                    'detections': d['detections']
                } for d in all_detections if d['alert']])
                
                # Cleanup
                try:
                    os.unlink(video_path)
                except:
                    pass

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