# 🚨 Fall Detection System

AI-powered fall detection system using YOLOv8 and Streamlit. Detect falls in images and videos with high accuracy.

## 🌟 Features

- ✅ **Image Detection**: Upload images for instant fall detection
- ✅ **Video Processing**: Process videos frame-by-frame with progress tracking
- ✅ **Real-time Alerts**: Get immediate notifications when falls are detected
- ✅ **Detailed Statistics**: View comprehensive detection metrics and logs
- ✅ **Easy to Use**: Simple web interface, no technical knowledge required
- ✅ **Cloud Ready**: Optimized for Streamlit Cloud deployment

## 🚀 Live Demo

👉 **[Try it now on Streamlit Cloud](https://your-app-url.streamlit.app)**

## 📋 How to Use

### 1. **Upload Your Model**
- Click on sidebar
- Upload your trained YOLOv8 model (`.pt` file)
- Set confidence threshold (0.5 recommended)

### 2. **Initialize Detector**
- Click "Initialize Detector" button
- Wait for model to load

### 3. **Upload Media**
- Go to "Detection" tab
- Choose Image or Video mode
- Upload your file

### 4. **View Results**
- For images: Click "Detect Falls"
- For videos: Click "Process Video"
- Download results and view detection logs

## 📦 Project Structure

```
fall-detection-app/
│
├── app.py                      # Main Streamlit application
├── utils/
│   └── fall_detector.py        # Fall detection logic
│
├── best_fall_model.pt          # Your trained model (optional)
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies
├── .streamlit/
│   └── config.toml            # Streamlit configuration
│
└── README.md                   # This file
```

## 🛠️ Local Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fall-detection-app.git
cd fall-detection-app
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the app**
```bash
streamlit run app.py
```

5. **Open browser**
- Navigate to `http://localhost:8501`

## ☁️ Deploy to Streamlit Cloud

### Step 1: Prepare Repository

1. Create new GitHub repository
2. Upload these files:
   - `app.py`
   - `utils/fall_detector.py`
   - `requirements.txt`
   - `packages.txt`
   - `README.md`
   - (Optional) `best_fall_model.pt` - if file is small

### Step 2: Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Select:
   - Repository: `your-username/fall-detection-app`
   - Branch: `main`
   - Main file path: `app.py`
5. Click "Deploy"!

### Step 3: Upload Model

Since model files are usually large (>100MB):

**Option A: Upload via App**
- Once deployed, use the file uploader in the sidebar
- Upload your `.pt` model file

**Option B: Use Git LFS** (for permanent model)
```bash
# Install Git LFS
git lfs install

# Track .pt files
git lfs track "*.pt"

# Add and commit
git add .gitattributes
git add best_fall_model.pt
git commit -m "Add model with LFS"
git push
```

**Option C: External Storage**
- Upload model to Google Drive, Dropbox, or AWS S3
- Modify `app.py` to download model on startup

## 🎯 Model Requirements

Your YOLOv8 model should:
- Have **2 classes**: `falling` and `normal`
- Be trained with YOLOv8 (v8.0+)
- Be in `.pt` format
- Support 640x640 input (recommended)

### Class Mapping
- **Class 0**: `falling` (person lying down/falling)
- **Class 1**: `normal` (person standing/sitting)

## 📊 Model Training

### Dataset Preparation (Roboflow)
1. Create project in Roboflow
2. Upload images with annotations
3. Classes: "falling" and "normal"
4. Export as "YOLOv8" format

### Training (Google Colab)
```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='path/to/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='fall_detection'
)

# Export
model.export(format='pt')
```

## 🔧 Configuration

### Adjust Confidence Threshold
- Lower (0.3-0.5): More detections, may have false positives
- Medium (0.5-0.7): Balanced (recommended)
- Higher (0.7-0.9): Fewer detections, high precision

### Video Processing
- Process every N frames: Higher = faster, lower = more accurate
- Show preview: Enable to see real-time processing

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support:
- Open an issue on GitHub
- Email: your.email@example.com

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Streamlit framework
- OpenCV library

---

Made with ❤️ using Streamlit and YOLOv8