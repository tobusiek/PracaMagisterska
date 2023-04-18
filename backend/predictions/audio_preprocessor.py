from typing import Callable

import librosa
from librosa.effects import harmonic, trim
from librosa.feature import (
    chroma_stft, rms, spectral_centroid, spectral_bandwidth,
    spectral_rolloff, zero_crossing_rate, tempo, mfcc
)
import numpy as np
import pandas as pd

from .temp_file_creator import TempFileCreator
from tools.const_variables import DATASET_INFO


class AudioPreprocessor:
    _temp_file_creator = TempFileCreator()
    _dataset_column_names: list[str] = DATASET_INFO['features']
    _length_of_dataset_records = DATASET_INFO['length_of_records']
    
    _features_without_mfcc_and_tempo: list[Callable[[np.ndarray], float]] = [
        chroma_stft, rms, spectral_centroid, spectral_bandwidth,
        spectral_rolloff, zero_crossing_rate, harmonic
        ]

    def preprocess_audio(self, request_id: str, file_data: bytes, file_extension: str) -> pd.DataFrame:
        temp_file = self._temp_file_creator.create_temp_file(request_id, file_data, file_extension)
        audio, sr = librosa.load(temp_file)
        audio = self._trim_audio(audio)
        audio_matrix = self._create_audio_matrix(audio)
        audio_df = self._create_dataframe(audio_matrix)
        audio_df = self._minmax_audio_df(audio_df)
        audio_df = self._convert_audio_df_to_float32(audio_df)
        self._temp_file_creator.delete_temp_file(request_id)
        return audio_df
    
    @staticmethod
    def _trim_audio(audio: np.ndarray) -> np.ndarray:
        return trim(audio)[0]
    
    @staticmethod
    def _get_mean_and_var(feature: np.ndarray) -> tuple[float, float]:
        return np.mean(feature), np.var(feature)
    
    def _get_features_for_row(self, split: np.ndarray) -> np.ndarray:
        features_for_row = []
        for feature in self._features_without_mfcc_and_tempo:
            feature_mean, feature_var = self._get_mean_and_var(feature(y=split))
            features_for_row.append(feature_mean)
            features_for_row.append(feature_var)
        features_for_row.append(tempo(y=split))
        for mfcc_ in mfcc(y=split):
            mfcc_mean, mfcc_var = self._get_mean_and_var(mfcc_)
            features_for_row.append(mfcc_mean)
            features_for_row.append(mfcc_var)
        return features_for_row
    
    def _split_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        n_splits = len(audio) // self._length_of_dataset_records
        return np.array_split(audio, n_splits)
    
    def _trim_split(self, split: np.ndarray) -> np.ndarray:
        return split[:self._length_of_dataset_records]
    
    def _create_audio_matrix(self, audio: np.ndarray) -> list[np.ndarray]:
        audio_matrix: list[np.ndarray] = []
        audio_splits = self._split_audio(audio)
        for split in audio_splits:
            trimmed_split = self._trim_split(split)
            audio_matrix.append(self._get_features_for_row(trimmed_split))
        return audio_matrix
    
    def _create_dataframe(self, audio_matrix: list[np.ndarray]) -> pd.DataFrame:
        return pd.DataFrame(audio_matrix, columns=self._dataset_column_names)

    def _minmax_column(self, column: pd.Series, column_id: str) -> pd.DataFrame:
        column_info = self._dataset_info[column_id]
        column_min = column_info['min']
        column_max = column_info['max'] 
        minmaxed_column = (column - column_min) / (column_max - column_min)
        return minmaxed_column
    
    def _minmax_audio_df(self, audio_df: pd.DataFrame) -> pd.DataFrame:
        for column_id, column in enumerate(audio_df.columns):
            audio_df[column] = self._minmax_column(column, column_id)
        return audio_df
    
    def _convert_audio_df_to_float32(audio_df: pd.DataFrame) -> pd.DataFrame:
        for column in audio_df.columns:
            audio_df[column] = audio_df[column].astype(np.float32)
        return audio_df
