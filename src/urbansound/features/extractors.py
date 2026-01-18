from __future__ import annotations

import numpy as np
import librosa


def load_audio_fixed(path: str, sr: int = 22050, seconds: float = 4.0) -> tuple[np.ndarray, int]:

    y, sr = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * seconds)

    if len(y) < target_len:
        # Pad μέχρι target_len
        y = librosa.util.fix_length(y, size=target_len)
    else:
        # Trim μέχρι target_len
        y = y[:target_len]

    return y, sr


def mfcc_stats(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # (n_mfcc, T)

    feat = np.concatenate(
        [mfcc.mean(axis=1), mfcc.std(axis=1)],
        axis=0
    )

    return feat.astype(np.float32)
