from functools import partial
import logging
import multiprocessing as mp
from typing import Callable

from librosa.effects import harmonic
from librosa.feature import (
    chroma_stft, rms, spectral_centroid, spectral_bandwidth,
    spectral_rolloff, zero_crossing_rate, tempo, mfcc
)
import numpy as np
import pandas as pd

from predictions.audio_preprocessors.audio_exceptions import AudioTooShortException, AudioTooLongException
from predictions.audio_preprocessors.base_audio_preprocessor import BaseAudioPreprocessor
from predictions.temp_file_creator import TempFileCreator
from tools.const_variables import GTZAN_DATASET_INFO, CORES_TO_USE, MAX_AUDIO_DURATION

logger = logging.getLogger('preprocessor')


class AudioFeaturesPreprocessor(BaseAudioPreprocessor):
    """Class for audio features preprocessing."""

    _temp_file_creator = TempFileCreator(GTZAN_DATASET_INFO['audio_format'])
    _dataset_features: dict[str, dict[str, float]] = GTZAN_DATASET_INFO['features']
    _dataset_column_names: list[str] = list(_dataset_features.keys())
    _split_duration = GTZAN_DATASET_INFO['split_duration']
    _sampling_rate: int = GTZAN_DATASET_INFO['sampling_rate']
    _n_fft: int = GTZAN_DATASET_INFO['n_fft']
    _hop_length: int = GTZAN_DATASET_INFO['hop_length']

    def preprocess_audio(self, request_id: str, file_data: bytes, file_extension: str) -> np.ndarray:
        """Create temporary file from bytes, load it with librosa, trim it, make a dataframe, minmax features and delete temporary file."""

        audio = self._get_audio(request_id, file_data, file_extension)
        self._validate_audio_duration(request_id, audio)
        audio = self._trim_audio(audio)
        audio_matrix = self._create_audio_matrix(audio)
        audio_df = self._create_dataframe(audio_matrix)
        audio_df = self._minmax_audio_df(audio_df)
        audio_df = self._convert_audio_df_to_float32(audio_df)
        return audio_df.to_numpy()

    def _validate_audio_duration(self, request_id: str, audio: np.ndarray) -> None:
        """Validate if audio duration fits between split duration and max audio duration."""

        duration = self._get_audio_duration(audio)
        if duration < self._split_duration // self._sampling_rate:
            logger.warning(f'{request_id=} audio too short')
            raise AudioTooShortException(f'{request_id=} audio duration to short')
        if duration > MAX_AUDIO_DURATION:
            logger.warning(f'{request_id=} audio too long')
            raise AudioTooLongException(f'{request_id=} audio duration to long')
    
    def _create_audio_matrix(self, audio: np.ndarray) -> list[np.ndarray]:
        """Create matrix for audio, splitting it to fit length of dataset records."""

        logger.debug('extracting features...')
        audio_matrix: list[np.ndarray] = []
        audio_splits = self._split_audio(audio)
        with mp.Pool(CORES_TO_USE) as pool:
            for split in audio_splits:
                audio_matrix.append(pool.apply(self._get_features_for_split, args=(split,)))
            logger.debug('audio matrix created')
            return audio_matrix

    def _split_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        """Split the audio to fit length of dataset records."""

        n_splits = len(audio) // self._split_duration
        logger.debug(f'audio splitted to {n_splits} splits')
        return np.array_split(audio, n_splits)

    def _get_features_for_split(self, split: np.ndarray) -> list[float]:
        """Get features for split that were used in dataset."""

        trimmed_split = self._trim_split(split)
        features_for_row = []
        for feature in self._get_features_without_mfcc_and_tempo():
            feature_mean, feature_var = self._get_mean_and_var(feature(y=trimmed_split))
            features_for_row.append(feature_mean)
            features_for_row.append(feature_var)
        features_for_row.append(tempo(y=trimmed_split, hop_length=self._hop_length))
        for mfcc_ in mfcc(y=trimmed_split, sr=self._sampling_rate):
            mfcc_mean, mfcc_var = self._get_mean_and_var(mfcc_)
            features_for_row.append(mfcc_mean)
            features_for_row.append(mfcc_var)
        return features_for_row

    def _trim_split(self, split: np.ndarray) -> np.ndarray:
        return split[:self._split_duration]

    def _get_features_without_mfcc_and_tempo(self) -> list[Callable[[np.ndarray], np.ndarray]]:
        """Create feature functions with partially populated arguments."""

        sampling_rate = self._sampling_rate
        n_fft = self._n_fft
        hop_length = self._hop_length
        return [
            partial(chroma_stft, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length),
            partial(rms, hop_length=hop_length),
            partial(spectral_centroid, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length),
            partial(spectral_bandwidth, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length),
            partial(spectral_rolloff, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length),
            partial(zero_crossing_rate, frame_length=n_fft, hop_length=hop_length),
            harmonic
        ]

    @staticmethod
    def _get_mean_and_var(feature: np.ndarray) -> tuple[float, float]:
        """Get mean and var from feature."""

        return np.mean(feature), np.var(feature)
    
    def _create_dataframe(self, audio_matrix: list[np.ndarray]) -> pd.DataFrame:
        """Create dataframe for audio from matrix."""

        return pd.DataFrame(audio_matrix, columns=self._dataset_column_names)
    
    def _minmax_audio_df(self, audio_df: pd.DataFrame) -> pd.DataFrame:
        """Minmax columns in audio dataframe."""

        for column in audio_df.columns:
            audio_df[column] = self._minmax_column(audio_df[column], column)
        return audio_df

    def _minmax_column(self, column: pd.Series, column_id: str) -> pd.DataFrame:
        """Minmax column of audio dataframe."""

        column_info = self._dataset_features[column_id]
        column_min = column_info['min']
        column_max = column_info['max']
        minmaxed_column = (column - column_min) / (column_max - column_min)
        return minmaxed_column
    
    @staticmethod
    def _convert_audio_df_to_float32(audio_df: pd.DataFrame) -> pd.DataFrame:
        """Convert features in dataframe to float32."""

        for column in audio_df.columns:
            audio_df[column] = audio_df[column].astype(np.float32)
        return audio_df
