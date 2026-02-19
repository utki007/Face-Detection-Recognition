"""Data collection: capture training images from camera for face recognition."""
import cv2
import json
import os

from src.config import (
    CASCADE_PATH,
    DATASET_DIR,
    USERS_FILE,
    MAX_SAMPLES,
    VIDEO_DEVICE,
    EXIT_KEY,
)


def _assure_path_exists(path):
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)


def run():
    """Capture face images from camera and save to dataset."""
    vid_cam = cv2.VideoCapture(VIDEO_DEVICE)
    face_detector = cv2.CascadeClassifier(CASCADE_PATH)

    while True:
        try:
            face_id = int(input("Enter your unique ID: "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    name = input("Enter your name (or press Enter to skip): ").strip() or f"User {face_id}"

    # Update users.json
    _assure_path_exists(USERS_FILE)
    users = {}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            users = json.load(f)
    users[str(face_id)] = name
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)
    print(f"Registered as: {name}")

    count = 0
    _assure_path_exists(DATASET_DIR)

    print(f"\nCapturing images... Press '{chr(EXIT_KEY)}' to stop early, or wait for {MAX_SAMPLES} samples.\n")

    while True:
        _, image_frame = vid_cam.read()
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(image_frame, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
            count += 1
            img_path = os.path.join(DATASET_DIR, f"User.{face_id}.{count}.jpg")
            cv2.imwrite(img_path, gray[y : y + h, x : x + w])
            cv2.imshow("frame", image_frame)

        if cv2.waitKey(100) & 0xFF == EXIT_KEY or count >= MAX_SAMPLES:
            break

    vid_cam.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images for {name} (ID: {face_id}).")


if __name__ == "__main__":
    run()
