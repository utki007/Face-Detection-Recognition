# Face Recognition & Detection System

A Python-based face recognition and detection system using OpenCV and Local Binary Patterns Histograms (LBPH) for real-time facial identification and attendance tracking.

## Overview

This project implements a complete pipeline for face recognition:
1. **Data Collection** - Capture training images from camera
2. **Model Training** - Train LBPH recognizer on collected images
3. **Face Recognition** - Real-time face detection and identification from video stream
4. **Attendance Tracking** - Track recognized individuals

## Features

- Real-time face detection using Haar Cascade classifiers
- Face recognition using LBPH (Local Binary Patterns Histograms) algorithm
- Confidence scoring for predictions
- Attendance tracking system
- Easy-to-use command-line interface

## Project Structure

```
.
├── face_datasets.py              # Data collection script
├── training.py                   # Model training script
├── face_recognition.py           # Real-time recognition script
├── haarcascade_frontalface_default.xml  # Pre-trained Haar Cascade
├── dataSet/                      # Training image dataset
├── trainer/                      # Trained model storage
├── Classifiers/                  # Additional classifiers
└── requirements.txt              # Python dependencies
```

## Prerequisites

- Python 3.6+
- Webcam/Camera
- macOS, Linux, or Windows

### System Permissions (macOS)

On macOS, you must grant camera permissions:
1. Go to **System Preferences > Security & Privacy > Camera**
2. Add Python/Terminal to the allowed applications

## Installation

### 1. Clone or Download the Project

```bash
cd "Face Recognition & Detection/Implementation"
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- opencv-contrib-python
- numpy
- pillow
- pandas

## Usage

### Step 1: Collect Training Data

Run the data collection script to capture facial images:

```bash
python face_datasets.py
```

**Process:**
1. Enter a unique integer ID for the person (e.g., 1, 2, 3)
2. Look at the camera and move naturally
3. The system will capture ~100+ images automatically and save them as:
   - `dataSet/User.{id}.{count}.jpg`
4. Press **'q'** to stop the session

**Notes:**
- Ensure good lighting conditions
- Capture different angles and expressions
- Collect data for multiple individuals (at least 2-3 people)

### Step 2: Train the Model

After collecting sufficient data, train the recognition model:

```bash
python training.py
```

**Output:**
- Trains on images in the `dataSet/` folder
- Saves trained model to `trainer/trainer.yml`
- Displays number of samples and individuals processed

### Step 3: Run Face Recognition

Launch the real-time face recognition system:

```bash
python face_recognition.py
```

**Features:**
- Real-time video feed with face detection
- Green rectangles around detected faces
- Displays recognized person's name and confidence score
- Tracks attendance for each recognized individual
- Press **'q'** to exit

## File Descriptions

### `face_datasets.py`
Captures training images from the camera and stores them in the dataset folder.
- Opens camera (device 0)
- Detects faces using Haar Cascade
- Saves cropped face images with format: `User.{id}.{count}.jpg`

### `training.py`
Trains the LBPH face recognizer using collected images.
- Loads images from `dataSet/` folder
- Extracts faces from training images
- Creates and trains LBPH model
- Saves model to `trainer/trainer.yml`

### `face_recognition.py`
Real-time face recognition and identification system.
- Loads trained model from `trainer/trainer.yml`
- Captures live video from camera
- Detects and identifies faces
- Displays confidence scores (lower is better, <100 is positive match)
- Maintains attendance array for tracked individuals

### `haarcascade_frontalface_default.xml`
Pre-trained Haar Cascade classifier for frontal face detection provided by OpenCV.

## Configuration

### Image Naming Convention
Images must follow this naming pattern: `User.{id}.{count}.jpg`
- `{id}` = unique person identifier (integer)
- `{count}` = image number (incremented for each capture)
- Example: `User.1.1.jpg`, `User.1.2.jpg`, `User.2.1.jpg`

### Recognition Parameters

In `face_recognition.py`, you can adjust:
- **Confidence threshold** (line ~50): `if confidence<100:`
  - Lower values = stricter matching
  - Higher values = more lenient
- **Face detection scale** (line ~37): `scaleFactor=1.2`
  - Adjust sensitivity of face detection

### User IDs

The system currently recognizes:
- ID 35 → "Utkarsh"
- ID 2 → "Mohit"
- ID 39 → "Rony"
- ID 4 → "MANAN"

To add more users, modify the ID mapping in `face_recognition.py` (around line 55-66).

## Troubleshooting

### Camera Not Opening

**Problem:** "Could not open camera"

**Solutions:**
1. Ensure camera permissions are granted (especially on macOS)
2. Check if another application is using the camera
3. Try a different camera device by changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`
4. Reconnect your camera

### Training Fails / No Faces Found

**Problem:** "No faces detected in images"

**Solutions:**
1. Ensure images are in `dataSet/` folder (note: capital S)
2. Check image quality and lighting
3. Ensure faces are clearly visible and at least 100x100 pixels
4. Re-capture training data with better lighting and angles

### Recognition Not Working

**Problem:** Faces not recognized or all marked as "Unknown"

**Solutions:**
1. Verify `trainer/trainer.yml` exists
2. Ensure model was trained on sufficient data (>50 images per person)
3. Lower the confidence threshold in `face_recognition.py`
4. Re-train the model with more varied images
5. Check that person IDs in `face_recognition.py` match your training data IDs

### Quit Key Not Working

**Problem:** Can't exit by pressing 'q'

**Solutions:**
1. Click on the video window to ensure it has focus
2. Try pressing 'q' multiple times
3. Press `Ctrl+C` in the terminal to force exit

## Algorithm Details

### Haar Cascade Classifier
- Uses cascade of trained classifiers
- Efficient, real-time face detection
- Pre-trained on frontal faces

### LBPH (Local Binary Patterns Histograms)
- Robust face recognition algorithm
- Creates histograms of local binary patterns
- Good performance with minimal training data
- Built into `cv2.face.LBPHFaceRecognizer`

## Performance Tips

1. **Lighting:** Use consistent, good lighting when capturing training images
2. **Variety:** Capture faces from different angles and distances
3. **Quantity:** More training images = better accuracy
4. **Cleanliness:** Keep camera lens clean
5. **Distance:** Keep faces 1-3 feet from camera

## Limitations

- Works best with frontal faces
- Requires good lighting conditions
- Limited to individuals trained in the model
- Does not handle occlusions (glasses, masks) well
- Single-threaded (may lag with many faces)

## Future Improvements

- Support for side-profile and rotated faces
- Deep learning models (CNN-based recognition)
- Multi-threading for better performance
- Web interface for easier management
- Database for storing attendance records
- Support for face masks and accessories

## License

This project uses OpenCV which is released under the BSD license.

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [Face Recognition with OpenCV](https://docs.opencv.org/4.5.2/dd/d65/classcv_1_1face_1_1LBPHFaceRecognizer.html)
- [Haar Cascade Classifiers](https://docs.opencv.org/4.5.2/db/d28/tutorial_cascade_classifier.html)

## Author

Utkarsh Narain - Face Recognition & Detection Project

## Support

For issues or questions, ensure:
1. All dependencies are installed: `pip install -r requirements.txt`
2. Camera permissions are granted
3. Follow the usage steps in order (collect → train → recognize)
4. Check the troubleshooting section above
