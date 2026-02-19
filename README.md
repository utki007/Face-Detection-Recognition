# Face Recognition & Detection System

A Python pipeline for face detection and recognition using OpenCV. Haar Cascade for detection, LBPH for recognition.

**Quick start:** `pip install -r requirements.txt` → `python main.py`

**Menu:** 1) Enroll faces (collect + train) 2) Face recognition 3) Exit. Press `q` to exit camera windows.

---

## What the Code Does

### Two-Stage Pipeline

1. **Detection** — Locate face regions in each frame with `cv2.CascadeClassifier` (Haar).
2. **Recognition** — Identify each detected face with `cv2.face.LBPHFaceRecognizer`, which outputs a numeric ID and a confidence value (lower = better match).

The recognizer is trained on grayscale face crops. IDs are integers; names come from `config/users.json`.

---

## Data Flow

```
Camera → Grayscale frame → Haar Cascade detects faces → For each face:
  → Crop face region → LBPH predict(ID, confidence) → Map ID to name via users.json
  → Draw box + label on frame → Display
```

---

## Module Breakdown

### `src/config.py`

Defines paths and constants. All paths are relative to the project root.

- **Paths**: `data/dataset/` (training images), `models/` (trained LBPH model), `config/users.json` (ID→name mapping), `assets/` (Haar cascade XML).
- **Detection**: `scaleFactor=1.2`, `minNeighbors=5` for `detectMultiScale`.
- **Recognition**: `CONFIDENCE_THRESHOLD=100` — LBPH returns lower values for better matches; values below this are treated as a valid identification.

---

### `src/collect.py` — Data Collection

**Purpose**: Capture face crops from the camera and write them to disk.

**Process**:
1. Prompts for a numeric `face_id` and optional name.
2. Writes `{face_id: name}` to `config/users.json`.
3. Opens the camera with `cv2.VideoCapture`.
4. For each frame:
   - Converts to grayscale.
   - Runs `face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))`.
   - For each bounding box `(x, y, w, h)`, crops `gray[y:y+h, x:x+w]` and saves as `data/dataset/User.{face_id}.{count}.jpg`.
5. Stops after `MAX_SAMPLES` (20) images or when the user presses the exit key.

**Output**: Grayscale face images named `User.{id}.{count}.jpg`. The ID in the filename is the label used during training.

---

### `src/train.py` — Model Training

**Purpose**: Train the LBPH recognizer on images in `data/dataset/` and save the model.

**Process**:
1. Scans `data/dataset/` for `.jpg`, `.jpeg`, `.png`.
2. Extracts the user ID from the filename (`User.{id}.{count}.jpg` → `id`).
3. Loads each image, converts to grayscale (PIL then numpy), runs the Haar cascade to find faces, and crops each face. Each crop is paired with its ID.
4. Builds two arrays: `face_samples` (cropped grayscale images) and `ids` (integer labels).
5. Calls `recognizer.train(faces, np.array(ids))` where `recognizer = cv2.face.LBPHFaceRecognizer_create()`.
6. Saves the model to `models/trainer.yml`.
7. Updates `config/users.json`: adds any new IDs from the dataset with default name `"User {id}"`, keeps existing names.

**Output**: `models/trainer.yml` (LBPH model) and an updated `config/users.json`.

---

### `src/recognize.py` — Recognition

**Purpose**: Run live face recognition from the camera.

**Process**:
1. Loads `config/users.json` into a `users` dict.
2. Loads the LBPH model from `models/trainer.yml`.
3. Opens the camera and runs a loop:
   - Reads a frame, converts to grayscale.
   - Runs `face_cascade.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBORS)`.
   - For each detected face `(x, y, w, h)`:
     - Crops `gray[y:y+h, x:x+w]`.
     - Calls `recognizer.predict(crop)` → `(face_id, confidence)`.
     - If `confidence < CONFIDENCE_THRESHOLD`: maps `face_id` to name via `users`, displays `{name} {100-confidence:.2f}%`.
     - Else: displays `"Unknown {100-confidence:.2f}%"`.
   - Draws a green rectangle and label on the frame.


---

### `main.py`

Provides a 3-option CLI menu:
1. **Enroll faces** — runs `collect.run()` then `train.run()` (capture images, then train on the full dataset)
2. **Face recognition** — runs `recognize.run()`
3. **Exit**

---

## Algorithms

### Haar Cascade (`cv2.CascadeClassifier`)

Pre-trained XML classifier. `detectMultiScale` slides a window over the image at different scales and applies a cascade of simple tests to reject non-face regions quickly. Returns bounding boxes `(x, y, w, h)` for each face.

### LBPH (`cv2.face.LBPHFaceRecognizer`)

1. Splits each face into small cells.
2. Computes Local Binary Pattern (LBP) for each cell — compares each pixel to its neighbors to form a binary code.
3. Builds a histogram of LBPs per cell.
4. Concatenates histograms into one feature vector per face.
5. At prediction time: compares the query face’s feature vector to stored vectors, returns the closest ID and a distance (confidence). Lower distance = better match.

---

## Project Structure

```
.
├── main.py                 # CLI entry point
├── src/
│   ├── config.py           # Paths, thresholds, camera index
│   ├── collect.py          # Camera → face crops → dataset
│   ├── train.py            # Dataset → LBPH model
│   └── recognize.py        # Camera → detection → recognition
├── data/dataset/           # User.{id}.{count}.jpg (grayscale face crops)
├── models/                 # trainer.yml (serialized LBPH)
├── config/users.json       # {"id": "name"}
└── assets/                 # haarcascade_frontalface_default.xml
```

---

## Run

```bash
pip install -r requirements.txt
python main.py
```

**Prerequisites:** Python 3.6+, webcam. Grant camera permissions on macOS (System Preferences → Security & Privacy → Camera).

Run modules directly (without the menu):
```bash
python -m src.collect
python -m src.train
python -m src.recognize
```

---

## Dependencies

- **opencv-contrib-python** — Haar cascade, LBPH, video capture
- **numpy** — Array operations
- **pillow** — Image loading in `train.py`
