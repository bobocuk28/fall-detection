# app.py - STREAMLIT CLOUD VERSION (Video & Image Upload)
import streamlit as st
import tempfile
import os
from PIL import Image
import numpy as np
import time
from datetime import datetime

# ============================================
# PAGE CONFIG (HARUS DI AWAL)
# ============================================
st.set_page_config(
    page_title="Fall Detection AI",
    page_icon="🚨",
    layout="wide"
)

# ============================================
# TITLE
# ============================================
st.title("🚨 AI Fall Detection System")
st.markdown("**University Final Project - Cloud Deployment**")

# ============================================
# CHECK FOR AI MODEL
# ============================================
try:
    from ultralytics import YOLO
    import cv2
    AI_AVAILABLE = True
    
    # Load model (gunakan model yang sudah ada)
    try:
        model = YOLO('best_fall_model.pt')  # Model custom
        MODEL_LOADED = True
    except:
        # Fallback ke model default
        model = YOLO('yolov8n.pt')
        MODEL_LOADED = False
        
except ImportError:
    AI_AVAILABLE = False
    st.warning("⚠️ AI Model not available. Running in simulation mode.")

# ============================================
# SESSION STATE
# ============================================
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'current_file' not in st.session_state:
    st.session_state.current_file = None

# ============================================
# SIDEBAR - SYSTEM STATUS
# ============================================
with st.sidebar:
    st.header("🛠️ System Status")
    
    # Dependencies status
    if AI_AVAILABLE:
        st.success("✅ AI Engine: Available")
        if MODEL_LOADED:
            st.success("✅ Fall Detection Model: Loaded")
        else:
            st.warning("⚠️ Using Default Model")
    else:
        st.error("❌ AI Engine: Not Available")
    
    st.divider()
    
    # Mode info
    st.subheader("📋 Available Features:")
    st.markdown("""
    - ✅ Image Upload & Analysis
    - ✅ Video Upload & Analysis  
    - ❌ Webcam (Not in Cloud)
    - ✅ Real-time Processing
    """)
    
    st.divider()
    
    # Instructions
    st.subheader("📖 Instructions:")
    st.markdown("""
    1. Upload image/video
    2. Click Analyze
    3. View results
    4. Download if needed
    """)

# ============================================
# MAIN CONTENT - UPLOAD SECTION
# ============================================
st.header("📁 Upload & Analyze")

# File type selection
file_type = st.radio(
    "Select file type to upload:",
    ["Image (JPG, PNG)", "Video (MP4, AVI, MOV)"],
    horizontal=True
)

# File uploader based on type
if "Image" in file_type:
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload JPG, JPEG, or PNG files up to 200MB"
    )
    
    if uploaded_file:
        # Display image preview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
        with col2:
            st.subheader("📊 File Info")
            st.write(f"**Name:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.write(f"**Type:** {uploaded_file.type}")
            st.write(f"**Dimensions:** {image.size}")
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                st.session_state.processing = True
                st.session_state.current_file = uploaded_file
                st.rerun()
                
elif "Video" in file_type:
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload MP4, AVI, MOV, or MKV files up to 200MB"
    )
    
    if uploaded_file:
        # Display video preview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Save temporarily to display video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
                
            # Show video player
            st.video(tmp_path)
            
        with col2:
            st.subheader("📊 File Info")
            st.write(f"**Name:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / (1024*1024):.1f} MB")
            st.write(f"**Type:** {uploaded_file.type}")
            
            # Try to get video info
            try:
                if AI_AVAILABLE:
                    cap = cv2.VideoCapture(tmp_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frames / fps if fps > 0 else 0
                    cap.release()
                    
                    st.write(f"**Duration:** {duration:.1f} sec")
                    st.write(f"**FPS:** {fps:.1f}")
            except:
                pass
            
            # Analyze button
            if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
                st.session_state.processing = True
                st.session_state.current_file = uploaded_file
                st.rerun()
        
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

# ============================================
# PROCESSING SECTION
# ============================================
if st.session_state.processing and st.session_state.current_file:
    st.divider()
    st.header("🔬 Processing Results")
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Determine file type
    is_image = st.session_state.current_file.type.startswith('image/')
    
    if is_image:
        # Process image
        with st.spinner("Analyzing image..."):
            for i in range(1, 101, 20):
                progress_bar.progress(i / 100)
                status_text.text(f"Processing... {i}%")
                time.sleep(0.2)
            
            # Get results
            if AI_AVAILABLE:
                # Real AI processing
                try:
                    # Load image
                    image = Image.open(st.session_state.current_file)
                    img_array = np.array(image)
                    
                    # Run detection
                    results = model(img_array)
                    result = results[0]
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📈 Detection Results")
                        
                        if result.boxes is not None:
                            boxes = result.boxes.cpu().numpy()
                            for i in range(len(boxes)):
                                cls_id = int(boxes.cls[i])
                                conf = boxes.conf[i]
                                class_name = model.names[cls_id]
                                
                                st.metric(
                                    label=f"Detection {i+1}",
                                    value=class_name.upper(),
                                    delta=f"{conf:.1%} confidence"
                                )
                        else:
                            st.info("No objects detected")
                    
                    with col2:
                        st.subheader("🖼️ Processed Image")
                        if len(results) > 0:
                            # Plot results
                            plotted = results[0].plot()
                            st.image(plotted, caption="AI Detection", use_container_width=True)
                            
                            # Download button
                            result_img = Image.fromarray(plotted[..., ::-1])  # BGR to RGB
                            from io import BytesIO
                            buf = BytesIO()
                            result_img.save(buf, format="PNG")
                            
                            st.download_button(
                                label="📥 Download Result",
                                data=buf.getvalue(),
                                file_name=f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png"
                            )
                    
                except Exception as e:
                    st.error(f"AI Processing Error: {e}")
                    # Fallback to simulation
                    st.info("Running in simulation mode due to error")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Person", "Detected", "✓")
                    with col2:
                        st.metric("Status", "Normal", "Safe")
                    with col3:
                        st.metric("Confidence", "92%", "High")
                    
                    st.image(image, caption="Original Image", use_container_width=True)
            else:
                # Simulation mode
                st.info("🎭 Simulation Mode Active")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Person", "Detected", "✓")
                with col2:
                    st.metric("Status", "Normal", "Safe")
                with col3:
                    st.metric("Confidence", "94%", "High")
                
                st.image(image, caption="Simulated Result", use_container_width=True)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis Complete!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
    
    else:
        # Process video
        with st.spinner("Analyzing video frames..."):
            # Save video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(st.session_state.current_file.getvalue())
                tmp_path = tmp_file.name
            
            if AI_AVAILABLE:
                try:
                    # Process video with AI
                    cap = cv2.VideoCapture(tmp_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    st.info(f"Processing {total_frames} frames at {fps:.1f} FPS")
                    
                    # Create placeholders for results
                    result_placeholder = st.empty()
                    stats_placeholder = st.empty()
                    
                    # Process frames
                    fall_count = 0
                    processed_frames = 0
                    sample_frames = []
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        processed_frames += 1
                        
                        # Update progress
                        progress = processed_frames / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {processed_frames}/{total_frames}")
                        
                        # Sample some frames for display
                        if processed_frames % 30 == 0:  # Every 30th frame
                            # Run detection
                            results = model(frame)
                            result_frame = results[0].plot()
                            sample_frames.append(result_frame)
                            
                            # Check for falls
                            if results[0].boxes is not None:
                                for box in results[0].boxes:
                                    cls_id = int(box.cls[0])
                                    if model.names[cls_id] == 'falling':
                                        fall_count += 1
                    
                    cap.release()
                    
                    # Display results
                    st.success(f"✅ Video Analysis Complete!")
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Frames", total_frames)
                    with col2:
                        st.metric("Processed", processed_frames)
                    with col3:
                        st.metric("Fall Detections", fall_count)
                    with col4:
                        st.metric("Fall Ratio", f"{(fall_count/processed_frames*100):.1f}%")
                    
                    # Show sample frames
                    if sample_frames:
                        st.subheader("📸 Sample Detections")
                        cols = st.columns(min(3, len(sample_frames)))
                        for idx, frame in enumerate(sample_frames[:3]):
                            with cols[idx]:
                                st.image(frame, caption=f"Frame {idx*30}", use_container_width=True)
                    
                    # Download results button
                    st.download_button(
                        label="📥 Download Analysis Report",
                        data=f"""
                        Fall Detection Analysis Report
                        =============================
                        File: {st.session_state.current_file.name}
                        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        
                        Statistics:
                        - Total Frames: {total_frames}
                        - Processed Frames: {processed_frames}
                        - Fall Detections: {fall_count}
                        - Fall Ratio: {(fall_count/processed_frames*100):.1f}%
                        
                        Analysis Complete.
                        """,
                        file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Video Processing Error: {e}")
                    # Fallback to simulation
                    st.info("Running video simulation due to error")
                    
                    # Simulated video results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Duration", "15.2s", "✓")
                    with col2:
                        st.metric("Falls Detected", "0", "Safe")
                    with col3:
                        st.metric("Risk Level", "Low", "Good")
                    
                    st.info("Simulated: No falls detected in video")
            
            else:
                # Simulation mode for video
                progress_bar.progress(50)
                status_text.text("Simulating video analysis...")
                time.sleep(2)
                
                st.success("✅ Video Analysis Complete!")
                
                # Simulated results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Frames", "456")
                with col2:
                    st.metric("Processing Time", "8.2s")
                with col3:
                    st.metric("Falls Detected", "1", "⚠️ Alert")
                with col4:
                    st.metric("Risk Level", "Medium")
                
                st.warning("Simulation: 1 fall detected at 00:12")
            
            # Clean up
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis Complete!")
            time.sleep(0.5)
    
    # Reset button
    if st.button("🔄 Analyze Another File", type="secondary"):
        st.session_state.processing = False
        st.session_state.current_file = None
        st.rerun()

# ============================================
# HISTORY SECTION
# ============================================
if not st.session_state.processing:
    st.divider()
    st.header("📋 Analysis History")
    
    # Sample history (in real app, you would store this)
    sample_history = [
        {"type": "Image", "name": "person_falling.jpg", "date": "2024-01-15 14:30", "result": "Fall Detected", "confidence": "87%"},
        {"type": "Video", "name": "elderly_monitoring.mp4", "date": "2024-01-15 14:15", "result": "Normal", "confidence": "92%"},
        {"type": "Image", "name": "standing_person.png", "date": "2024-01-15 13:45", "result": "Normal", "confidence": "95%"},
    ]
    
    for item in sample_history:
        with st.container():
            cols = st.columns([1, 2, 2, 2, 1])
            with cols[0]:
                st.write("🖼️" if item["type"] == "Image" else "🎥")
            with cols[1]:
                st.write(item["name"])
            with cols[2]:
                st.write(item["date"])
            with cols[3]:
                if "Fall" in item["result"]:
                    st.error(item["result"])
                else:
                    st.success(item["result"])
            with cols[4]:
                st.write(item["confidence"])
            st.divider()

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🎓 <b>Fall Detection AI System</b> - University Final Project</p>
    <p>Deployed on Streamlit Cloud | Upload & Analyze Images/Videos</p>
</div>
""", unsafe_allow_html=True)