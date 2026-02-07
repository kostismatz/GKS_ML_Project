import os.path

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost.testing.data import joblib

from cache_manager import CacheManager, create_result_directories
from config.config import Config
from data_loader import DataLoader
from feature_extraction import FeatureExtractor
from models import ModelFactory


class Trainer:
    def __init__(self, feature_set="baseline"):
        self.config = Config()

        self.feature_set = feature_set
        self.loader = DataLoader()
        from pre_processing import PreProcessor
        self.preprocessor = PreProcessor()
        self.feature_extractor = FeatureExtractor()
        self.cache_manager = CacheManager()
        self.model_factory = ModelFactory()

        create_result_directories()

    def extract_features_from_dataset(self, dataframe):

        X = []
        Y = []

        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            audio_path = self.loader.get_audio_path(row["slice_file_name"], row['fold'])

            features = self.cache_manager.load_cached_features(audio_path, self.feature_set)

            if features is None:
                audio, sr = self.preprocessor.load_and_preprocess(audio_path)
                features = self.feature_extractor.extract_features(audio, sr)
                self.cache_manager.save_cached_features(audio_path, self.feature_set, features)

            X.append(features)
            Y.append(row["classID"])
        return np.array(X), np.array(Y)

    def train_model(self, model_name, X_train, Y_train, use_csv=True):

        model = self.model_factory.get_model(model_name)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        cv_scores = None

        if use_csv:
            cv_scores = cross_val_score(
                model,
                X_train_scaled,
                Y_train,
                cv=self.config.CV_FOLDS,
                scoring="accuracy",
                n_jobs=-1
            )
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        model.fit(X_train_scaled, Y_train)
        self.save_model(model, scaler, model_name)

        return model, scaler, cv_scores

    def get_tag(self, model_name):
        return f"{model_name}_{self.feature_set}"

    def save_model(self, model, scaler, model_name):
        tag = self.get_tag(model_name)
        model_path = os.path.join(self.config.MODELS_DIR, f"{tag}_model.pkl")
        scaler_path = os.path.join(self.config.MODELS_DIR, f"{tag}_scaler.pkl")

        joblib.dump(scaler, model_path)
        joblib.dump(scaler, scaler_path)

        print(f"Model saved to {model_path}")


    def load_model(self, model_name):
        tag = self.get_tag(model_name)
        model_path = os.path.join(self.config.MODELS_DIR, f"{tag}_model.pkl")
        scaler_path = os.path.join(self.config.MODELS_DIR, f"{tag}_scaler.pkl")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        return model, scaler


