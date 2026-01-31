import os
import joblib
import hashlib
from typing import Optional

from config import config
from config.config import Config


class CacheManager:
    def __init__(self):
        self.config = Config()
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)

    def _feature_signature(self, feature_set: str) -> str:
        cache_version = getattr(self.config, "CACHE_VERSION", config.Config.CACHE_VERSION)

        return (
            f"{cache_version}|"
            f"set={feature_set}|"
            f"mfcc={self.config.N_MFCC}|"
            f"nfft={self.config.N_FFT}|"
            f"hop={self.config.HOP_LENGTH}"
        )

    def get_cache_path(self, file_path: str, feature_set: str) -> str:
        key = f"{file_path}|{self._feature_signature(feature_set)}"
        file_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
        cache_filename = f"{file_hash}.pkl"
        return os.path.join(self.config.CACHE_DIR, cache_filename)

    def load_cached_features(self, file_path: str, feature_set: str) -> Optional[object]:
        if not self.config.CACHE_FEATURES:
            return None

        cache_path = self.get_cache_path(file_path, feature_set)
        if not os.path.exists(cache_path):
            return None
        try:
            return joblib.load(cache_path)
        except Exception as e:
            print(f"Error loading cache {cache_path}: {e}")
            return None

    def save_cached_features(self, file_path: str, feature_set: str, features):
        if not self.config.CACHE_FEATURES:
            return
        cache_path = self.get_cache_path(file_path, feature_set)
        try:
            joblib.dump(features, cache_path)
        except Exception as e:
            print(f"Error saving cache {cache_path}: {e}")

    def clear_cache(self):
        cache_files = [f for f in os.listdir(self.config.CACHE_DIR) if f.endswith(".pkl")]
        for f in cache_files:
            try:
                os.remove(os.path.join(self.config.CACHE_DIR, f))
            except Exception as e:
                print(f"Error deleting cache file {f}: {e}")
        print(f"Cleared {len(cache_files)} cached files")


def create_result_directories():
    """Create all result directories if they don't exist."""
    config = Config()
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
