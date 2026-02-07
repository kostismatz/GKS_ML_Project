import os


class Config:

    # Project root directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Data paths
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'urbansound8k')
    PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed')

    # Results paths
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
    MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
    FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
    METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')

    TARGET_SR = 16000
    DURATION = 4.0

    N_MFCC = 13  # Number of MFCCs
    N_FFT = 2048  # FFT window size
    HOP_LENGTH = 512  # Number of samples between successive frames
    N_MELS = 128  # Number of mel bands

    CACHE_FEATURES = True
    CACHE_DIR = PROCESSED_DATA_PATH
    CACHE_VERSION = "v1"

    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5

    NUM_FOLDS = 10
    NUM_CLASSES = 10

    TEST_FOLD = 10

    CLASS_NAMES = [
        'air_conditioner',
        'car_horn',
        'children_playing',
        'dog_bark',
        'drilling',
        'engine_idling',
        'gun_shot',
        'jackhammer',
        'siren',
        'street_music'
    ]