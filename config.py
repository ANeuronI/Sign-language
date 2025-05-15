import os

# Debug flag
DEBUG = False

# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEOS_DIR = os.path.join(DATA_DIR, "vedios")  # Original directory name maintained
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
for directory in [DATA_DIR, VIDEOS_DIR, PROCESSED_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# File paths
CSV_FILE = os.path.join(DATA_DIR, "iSign_v1.1(1).csv")
PROCESSED_FRAMES_FILE = os.path.join(PROCESSED_DIR, "frames.npy")
PROCESSED_LANDMARKS_FILE = os.path.join(PROCESSED_DIR, "landmarks.npy")
PROCESSED_TEXT_FILE = os.path.join(PROCESSED_DIR, "text.npy")
TOKENIZER_FILE = os.path.join(PROCESSED_DIR, "tokenizer.pickle")
PREPROCESSED_DATA_FILE = os.path.join(PROCESSED_DIR, "preprocessed_data.npz")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "isl_translator_best.h5")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "isl_translator_final.h5")
ENCODER_MODEL_PATH = os.path.join(MODEL_DIR, 'encoder_model.h5')
DECODER_MODEL_PATH = os.path.join(MODEL_DIR, 'decoder_model.h5')

# Video processing parameters
MAX_FRAMES = 60  # Maximum frames to consider per video
FRAME_HEIGHT = 224
FRAME_WIDTH = 224

# Text processing parameters
MAX_SEQUENCE_LENGTH = 20  # Maximum sentence length

# Model parameters
BATCH_SIZE = 32
EPOCHS = 50
EMBEDDING_DIM = 256
LSTM_UNITS = 256
TEST_SIZE = 0.2  # For train/test split
RANDOM_SEED = 42  # For reproducibility