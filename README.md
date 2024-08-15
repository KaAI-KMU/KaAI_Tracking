# KaAI-DD Multimodal Object Tracking System

This repository contains the code and sample dataset for tracking objects using the KaAI-DD dataset. The system combines image and LiDAR data, leveraging a Kalman filter and the Hungarian algorithm for efficient and accurate real-time tracking.

## Features
- **Multimodal Data Integration:** Combines image and LiDAR data for robust object tracking.
- **Kalman Filter:** Used for estimating the position and velocity of tracked objects.
- **Hungarian Algorithm:** Ensures optimal matching of tracked objects using cost matrices.
- **ResNet-50:** Extracts feature vectors from images for improved visual tracking.

## 3D Object Tracking

### Instructions
1. **Set Up Paths:**
   - Modify the paths in the code:
     - `line 51`: Set to the path containing the annotated JSON files for the point cloud data.
     - `line 52`: Set to the path containing the image files (.png) from the forward camera (`flir4`).
     - `line 53`: Set to the path containing the LiDAR files (.pcd).
   
2. **Adjust Folder Name:**
   - Modify the folder name in `lines 207 and 208` to match the format `2024xxxx_drive_`, where `xxxx` should be adjusted accordingly.

3. **Run the Tracking Process:**
   - After making the necessary modifications, run the Python script in the terminal.

4. **Validate Results:**
   - Validate the tracking results using SustechPoint as per the annotation method described.

## 2D Object Tracking

### Instructions
1. **Set Up Path:**
   - Modify the path in `line 81` to point to the directory containing the 2D labels for the forward camera.

2. **Run the Tracking Process:**
   - Run the Python script in the terminal to complete the 2D tracking process.

---

### Notes
- Ensure all paths are correctly set up before running the tracking scripts.
- The system is designed for real-time object tracking and is optimized for use with the KaAI-DD dataset.
