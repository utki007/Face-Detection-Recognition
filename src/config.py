"""Shared configuration and paths for the Face Detection & Recognition system."""
import os

# Project root (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")

# Model paths
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "trainer.yml")

# Config paths
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
USERS_FILE = os.path.join(CONFIG_DIR, "users.json")

# Assets (Haar cascade, classifiers)
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
CASCADE_PATH = os.path.join(ASSETS_DIR, "haarcascade_frontalface_default.xml")

# Recognition parameters
CONFIDENCE_THRESHOLD = 100
SCALE_FACTOR = 1.2
MIN_NEIGHBORS = 5

# Data collection
MAX_SAMPLES = 50
VIDEO_DEVICE = 0
EXIT_KEY = ord("q")
