import os.path

import pandas as pd

from config.config import Config


class DataLoader:
    def __init__(self):
        self.config = Config()
        self.metadata_path = os.path.join(self.config.RAW_DATA_PATH, 'UrbanSound8K.csv')
        self.metadata = None

    def load_metadata(self):
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f'Metadata file not found: {self.metadata_path}')

        self.metadata = pd.read_csv(self.metadata_path)

        return self.metadata

    def get_audio_path(self, filename, fold):
        return os.path.join(self.config.RAW_DATA_PATH, f'fold{fold}', filename)

    def get_fold_data(self, fold_number):
        if self.metadata is None:
            self.load_metadata()
        fold_data = self.metadata[self.metadata['fold'] == fold_number]
        return fold_data

    def get_train_test_split(self, test_fold = 10):
        if self.metadata is None:
            self.load_metadata()

        train_df = self.metadata[self.metadata['fold'] != test_fold]
        test_df = self.metadata[self.metadata['fold'] == test_fold]

        return train_df, test_df

    def get_class_distribution(self):
        if self.metadata is None:
            self.load_metadata()

        return self.metadata['class'].value_counts()