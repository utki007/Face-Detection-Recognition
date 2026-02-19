"""Real-time face recognition from video stream."""
import cv2
import json
import os

from src.config import (
    CASCADE_PATH,
    USERS_FILE,
    MODEL_PATH,
    MODELS_DIR,
    CONFIDENCE_THRESHOLD,
    SCALE_FACTOR,
    MIN_NEIGHBORS,
    VIDEO_DEVICE,
    EXIT_KEY,
)


def _assure_path_exists(path):
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)


def run():
    """Run real-time face recognition."""
    _assure_path_exists(MODELS_DIR)

    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Run option 1 (Enroll faces) first.")
        return

    # Load users
    users = {}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            users = json.load(f)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cam = cv2.VideoCapture(VIDEO_DEVICE)

    print(f"\nFace recognition started. Press '{chr(EXIT_KEY)}' to exit.\n")

    while True:
        ret, im = cam.read()
        if not ret:
            continue

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBORS)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)

            face_id, confidence = recognizer.predict(gray[y : y + h, x : x + w])

            if confidence < CONFIDENCE_THRESHOLD:
                sid = str(face_id)
                name = users.get(sid, f"User {face_id}")
                label = f"{name} {100 - confidence:.2f}%"
            else:
                label = f"Unknown {100 - confidence:.2f}%"

            cv2.rectangle(im, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
            cv2.putText(im, label, (x, y - 40), font, 1, (255, 255, 255), 3)

        cv2.imshow("Face Recognition", im)

        if cv2.waitKey(10) & 0xFF == EXIT_KEY:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
