import logging
import multiprocessing as mp
from typing import Callable

import librosa
from librosa.effects import harmonic, trim
from librosa.feature import (
    chroma_stft, rms, spectral_centroid, spectral_bandwidth,
    spectral_rolloff, zero_crossing_rate, tempo, mfcc
)
import numpy as np
import pandas as pd

from predictions.temp_file_creator import TempFileCreator
from tools.const_variables import FMA_DATASET_INFO, CORES_TO_USE

logger = logging.getLogger('preprocessor')


class AudioFeaturesPreprocessor:
    '''Class for audio features preprocessing.'''

    _temp_file_creator = TempFileCreator()
    _dataset_features: dict[str, dict[str, float]] = FMA_DATASET_INFO['features']
    _dataset_column_names: list[str] = list(_dataset_features.keys())
    _length_of_dataset_records = FMA_DATASET_INFO['split_duration']
    
    _features_without_mfcc_and_tempo: list[Callable[[np.ndarray], float]] = [
        chroma_stft, rms, spectral_centroid, spectral_bandwidth,
        spectral_rolloff, zero_crossing_rate, harmonic
    ]

    def preprocess_audio(self, request_id: str, file_data: bytes, file_extension: str) -> np.ndarray:
        '''Create temporary file from bytes, load it with librosa, trim it, make a dataframe, minmax features and delete temporary file.'''

        logger.info(f'preprocessing audio for {request_id=}...')
        temp_file = self._temp_file_creator.create_temp_file(request_id, file_data, file_extension)
        audio, sr = librosa.load(temp_file)
        audio = self._trim_audio(audio)
        audio_matrix = self._create_audio_matrix(audio)
        audio_df = self._create_dataframe(audio_matrix)
        audio_df = self._minmax_audio_df(audio_df)
        audio_df = self._convert_audio_df_to_float32(audio_df)
        self._temp_file_creator.delete_temp_file(request_id)
        return audio_df.to_numpy()
    
    @staticmethod
    def _trim_audio(audio: np.ndarray) -> np.ndarray:
        '''Trim audio to get rid of silent parts.'''

        return trim(audio)[0]
    
    @staticmethod
    def _get_mean_and_var(feature: np.ndarray) -> tuple[float, float]:
        '''Get mean and var from feature.'''

        return np.mean(feature), np.var(feature)
    
    def _split_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        '''Split the audio to fit length of dataset records.'''

        n_splits = len(audio) // self._length_of_dataset_records
        logger.debug(f'audio splitted to {n_splits} splits')
        return np.array_split(audio, n_splits)
    
    def _trim_split(self, split: np.ndarray) -> np.ndarray:
        '''Trim the split to fit length of dataset records.'''

        return split[:self._length_of_dataset_records]
    
    def _get_features_for_split(self, split: np.ndarray) -> np.ndarray:
        '''Get features for split that were used in dataset.'''

        trimmed_split = self._trim_split(split)
        features_for_row = []
        for feature in self._features_without_mfcc_and_tempo:
            feature_mean, feature_var = self._get_mean_and_var(feature(y=trimmed_split))
            features_for_row.append(feature_mean)
            features_for_row.append(feature_var)
        features_for_row.append(tempo(y=trimmed_split))
        for mfcc_ in mfcc(y=trimmed_split):
            mfcc_mean, mfcc_var = self._get_mean_and_var(mfcc_)
            features_for_row.append(mfcc_mean)
            features_for_row.append(mfcc_var)
        return features_for_row
    
    def _create_audio_matrix(self, audio: np.ndarray) -> list[np.ndarray]:
        '''Create matrix for audio, splitting it to fit length of dataset records.'''

        logger.debug('extracting features...')
        audio_matrix: list[np.ndarray] = []
        audio_splits = self._split_audio(audio)
        with mp.Pool(CORES_TO_USE) as pool:
            for split in audio_splits:
                audio_matrix.append(pool.apply(self._get_features_for_split, args=(split,)))
            logger.debug('audio matrix created')
            return audio_matrix
    
    def _create_dataframe(self, audio_matrix: list[np.ndarray]) -> pd.DataFrame:
        '''Create dataframe for audio from matrix.'''

        return pd.DataFrame(audio_matrix, columns=self._dataset_column_names)

    def _minmax_column(self, column: pd.Series, column_id: str) -> pd.DataFrame:
        '''Minmax column of audio dataframe.'''

        column_info = self._dataset_features[column_id]
        column_min = column_info['min']
        column_max = column_info['max'] 
        minmaxed_column = (column - column_min) / (column_max - column_min)
        return minmaxed_column
    
    def _minmax_audio_df(self, audio_df: pd.DataFrame) -> pd.DataFrame:
        '''Minmax columns in audio dataframe.'''

        for column in audio_df.columns:
            audio_df[column] = self._minmax_column(audio_df[column], column)
        return audio_df
    
    @staticmethod
    def _convert_audio_df_to_float32(audio_df: pd.DataFrame) -> pd.DataFrame:
        '''Convert features in dataframe to float32.'''

        for column in audio_df.columns:
            audio_df[column] = audio_df[column].astype(np.float32)
        return audio_df
