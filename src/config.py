from pathlib import Path

# Configuration parameters for training and dataset processing
BATCH_SIZE = 4
EPOCHS = 5
IMAGE_SIZE = 256
LEARNING_RATE = 1e-4
THRESHOLD = 0.5
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
