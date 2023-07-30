import logging
import multiprocessing as mp

from librosa import zero_crossings
from librosa.beat import beat_track
from librosa.effects import hpss
from librosa.feature import (
    chroma_stft, rms, spectral_centroid, spectral_bandwidth,
    spectral_rolloff, mfcc
)
import numpy as np
import pandas as pd

from predictions.audio_preprocessors.audio_exceptions import AudioTooShortException, AudioTooLongException
from predictions.audio_preprocessors.base_audio_preprocessor import BaseAudioPreprocessor
from predictions.temp_file_creator import TempFileCreator
from tools.const_variables import GTZAN_DATASET_INFO, CORES_TO_USE, MAX_AUDIO_DURATION

logger = logging.getLogger('preprocessor')

FeatureMeanAndVar = tuple[float, float]


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
                trimmed_split = self._trim_split(split)
                audio_matrix.append(pool.apply(self._get_features_for_split, args=(trimmed_split,)))
            logger.debug('audio matrix created')
            return audio_matrix

    def _split_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        """Split the audio to fit length of dataset records."""

        n_splits = len(audio) // self._split_duration
        logger.debug(f'audio splitted to {n_splits} splits')
        return np.array_split(audio, n_splits)
    
    def _trim_split(self, split: np.ndarray) -> np.ndarray:
        """Trim the split to fit lenght of splits in dataset."""

        return split[:self._split_duration]

    def _get_features_for_split(self, split: np.ndarray) -> list[float]:
        """Get features for split that were used in dataset."""

        features_for_row = []
        features_for_row.extend(self._get_chroma_stft_features(split))
        features_for_row.extend(self._get_rms_features(split))
        features_for_row.extend(self._get_spectral_centroid_features(split))
        features_for_row.extend(self._get_spectral_bandwidth_features(split))
        features_for_row.extend(self._get_rolloff_features(split))
        features_for_row.extend(self._get_zcr_features(split))
        harmony_features, perceptr_features = self._get_harmony_and_perceptr_features(split)
        features_for_row.extend(harmony_features)
        features_for_row.extend(perceptr_features)
        features_for_row.append(self._get_tempo_feature(split))
        features_for_row.extend(self._get_mfcc_features(split))
        return features_for_row
    
    @staticmethod
    def _get_mean_and_var(feature: np.ndarray) -> FeatureMeanAndVar:
        """Get mean and var from feature."""

        return np.mean(feature), np.var(feature)
    
    def _get_chroma_stft_features(self, split: np.ndarray) -> FeatureMeanAndVar:
        chroma_stft_ = chroma_stft(y=split, sr=self._sampling_rate, hop_length=self._hop_length)
        return self._get_mean_and_var(chroma_stft_)
    
    def _get_rms_features(self, split: np.ndarray) -> FeatureMeanAndVar:
        rms_ = rms(y=split)
        return self._get_mean_and_var(rms_)
    
    def _get_spectral_centroid_features(self, split: np.ndarray) -> FeatureMeanAndVar:
        spectral_centroid_ = spectral_centroid(y=split, sr=self._sampling_rate)[0]
        return self._get_mean_and_var(spectral_centroid_)
    
    def _get_spectral_bandwidth_features(self, split: np.ndarray) -> FeatureMeanAndVar:
        spectral_bandwidth_ = spectral_bandwidth(y=split, sr=self._sampling_rate)
        return self._get_mean_and_var(spectral_bandwidth_)
    
    def _get_rolloff_features(self, split: np.ndarray) -> FeatureMeanAndVar:
        rolloff_ = spectral_rolloff(y=split, sr=self._sampling_rate)[0]
        return self._get_mean_and_var(rolloff_)
    
    def _get_zcr_features(self, split: np.ndarray) -> FeatureMeanAndVar:
        zcr = zero_crossings(y=split, pad=False)
        return self._get_mean_and_var(zcr)
    
    def _get_harmony_and_perceptr_features(self, split: np.ndarray) -> tuple[FeatureMeanAndVar, FeatureMeanAndVar]:
        harmony, perceptr = hpss(y=split)
        return self._get_mean_and_var(harmony), self._get_mean_and_var(perceptr)
    
    def _get_tempo_feature(self, split: np.ndarray) -> float:
        return beat_track(y=split, sr=self._sampling_rate)[0]
    
    def _get_mfcc_features(self, split: np.ndarray) -> list[FeatureMeanAndVar]:
        mfccs = []
        for mfcc_ in mfcc(y=split, sr=self._sampling_rate):
            mfccs.extend(self._get_mean_and_var(mfcc_))
        return mfccs
    
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
