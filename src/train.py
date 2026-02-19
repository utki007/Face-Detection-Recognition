"""Model training: train LBPH face recognizer on collected dataset."""
import cv2
import json
import os

import numpy as np
from PIL import Image

from src.config import CASCADE_PATH, DATASET_DIR, MODELS_DIR, USERS_FILE, MODEL_PATH


def _assure_path_exists(path):
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_images_and_labels(path):
    """Load face images and their IDs from the dataset directory."""
    if not os.path.exists(path):
        return [], []

    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    detector = cv2.CascadeClassifier(CASCADE_PATH)

    for image_path in image_paths:
        if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        PIL_img = Image.open(image_path).convert("L")
        img_numpy = np.array(PIL_img, "uint8")
        user_id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y : y + h, x : x + w])
            ids.append(user_id)

    return face_samples, ids


def run():
    """Train the LBPH recognizer and save the model."""
    _assure_path_exists(MODELS_DIR)

    faces, ids = get_images_and_labels(DATASET_DIR)

    if not faces or not ids:
        print("No face images found in dataset. Run data collection first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.save(MODEL_PATH)
    print(f"Trained on {len(faces)} samples from {len(set(ids))} person(s).")
    print(f"Model saved to {MODEL_PATH}")

    # Update users.json with IDs from dataset
    _assure_path_exists(USERS_FILE)
    unique_ids = sorted(set(ids))
    users = {}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            users = json.load(f)
    for uid in unique_ids:
        sid = str(uid)
        if sid not in users:
            users[sid] = f"User {uid}"
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)
    print("Updated users config with registered IDs.")


if __name__ == "__main__":
    run()
