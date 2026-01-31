import soundfile as sf
import numpy as np
import librosa
from config.config import Config


class PreProcessor:

    def __init__(self):
        self.config = Config()

    def load_and_preprocess(self, file_path):
        audio, sr = librosa.load(file_path, sr=None)

        if sr != self.config.TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.TARGET_SR)
            sr = self.config.TARGET_SR

        audio = self.normalize_audio(audio)
        audio = self.fix_length(audio, sr)

        return audio.astype(np.float32, copy=False), sr

    def normalize_audio(self, audio):
        eps = 1e-8
        max_val = np.max(np.abs(audio))
        if max_val > eps:
            audio = audio / max_val
        return audio

    def fix_length(self, audio, sr):

        target_length = int(self.config.DURATION * sr)

        if len(audio) < target_length:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        elif len(audio) > target_length:
            audio = audio[:target_length]

        return audio

    def get_sample_rate(self, file_path):
        info = sf.info(file_path)
        return info.samplerate
