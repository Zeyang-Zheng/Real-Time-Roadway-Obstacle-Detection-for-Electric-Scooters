# Real-Time Roadway Obstacle Detection for Electric Scooters Using Deep Learning and Multi-Sensor Fusion

The system uses the YOLOv5s model to detect ground obstacles. It combines RGB and depth images captured by the Intel RealSense Camera D435i to estimate the distance to ground obstacles in real time.

## 💻 Get Started
Linux

```
# Clone the repository:
git clone https://github.com/Zeyang-Zheng/Real-Time-Roadway-Obstacle-Detection-for-Electric-Scooters.git
cd Real-Time-Roadway-Obstacle-Detection-for-Electric-Scooters

# Create and active a conda environment
conda create -n e-scooters python=3.8
conda activate e-scooters

# Install dependencies:
pip install -r requirements.txt

# Run the code:
python detection_linux.py --view-img --conf-thres 0.5
```

Windows

```
# Clone the repository:
git clone https://github.com/Zeyang-Zheng/Real-Time-Roadway-Obstacle-Detection-for-Electric-Scooters.git
cd Real-Time-Roadway-Obstacle-Detection-for-Electric-Scooters

# Create and active a conda environment
conda create -n e-scooters python=3.8
conda activate e-scooters

# Install dependencies:
pip install -r requirements.txt

# Run the code:
python detection_windows.py --view-img --conf-thres 0.5
```

## 📖Citation
If you find this work is useful, consider citing the following article:

```

```

## 📚Reference
- YOLOv5 model: https://github.com/ultralytics/yolov5.

  
