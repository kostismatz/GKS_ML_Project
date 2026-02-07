import librosa
import librosa.feature as lf
import numpy as np

from config.config import Config


class FeatureExtractor:

    def __init__(self, feature_set: str = "baseline"):
        self.config = Config()
        self.feature_set = feature_set

    def extract_features(self, audio, sr):
        if self.feature_set == "baseline":
            return self._features_baseline(audio, sr)
        if self.feature_set == "xgb":
            return self._features_xgb(audio, sr)

        raise ValueError(f"Unknown feature set: {self.feature_set}")

    def _features_baseline(self, audio, sr):
        mfccs = self.extract_mfcc(audio, sr)
        spectral = self.extract_spectral_features(audio, sr)
        return np.concatenate((mfccs, spectral)).astype(np.float32)

    def _features_xgb(self, audio, sr):
        features = []

        mfcc = lf.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.config.N_MFCC,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH
        )
        features.extend(self._mean_std_per_row(mfcc).tolist())

        delta = lf.delta(mfcc, order=1)
        delta2 = lf.delta(mfcc, order=2)
        features.extend(self._mean_std_per_row(delta).tolist())
        features.extend(self._mean_std_per_row(delta2).tolist())

        features.extend(self.extract_spectral_features(audio, sr).tolist())

        mel = lf.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH
        )
        logmel = librosa.power_to_db(mel, ref=np.max)
        features.extend([float(np.mean(logmel)), float(np.std(logmel))])

        return np.array(features, dtype=np.float32)

    def extract_mfcc(self, audio, sr):
        mfcc = lf.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.config.N_MFCC,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH
        )

        return self._mean_std_per_row(mfcc).astype(np.float32)

    def extract_spectral_features(self, audio, sr):
        features = []

        centroid = lf.spectral_centroid(
            y=audio,
            sr=sr,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH
        )
        features.extend([np.mean(centroid), np.std(centroid)])

        rolloff = lf.spectral_rolloff(
            y=audio,
            sr=sr,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH
        )
        features.extend([np.mean(rolloff), np.std(rolloff)])

        zcr = lf.zero_crossing_rate(
            y=audio,
            frame_length=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH
        )
        features.extend([np.mean(zcr), np.std(zcr)])

        contrast = lf.spectral_contrast(
            y=audio,
            sr=sr,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH
        )
        features.extend(np.mean(contrast, axis=1).tolist())
        features.extend(np.std(contrast, axis=1).tolist())

        return np.array(features, dtype=np.float32)

    def _mean_std_per_row(self, mat):
        return np.concatenate([np.mean(mat, axis=1), np.std(mat, axis=1)])
