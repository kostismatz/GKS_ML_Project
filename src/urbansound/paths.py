from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# UrbanSound8K dataset paths
URBANSOUND_DIR = RAW_DIR / "UrbanSound8K"
METADATA_CSV = URBANSOUND_DIR / "metadata" / "UrbanSound8K.csv"
AUDIO_DIR = URBANSOUND_DIR / "audio"

# Outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "runs"

# Models
MODELS_DIR = PROJECT_ROOT / "models"
