from collections.abc import Generator
from functools import lru_cache
import os

import keras.utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from keras.callbacks import EarlyStopping
import librosa
import numpy as np
import pandas as pd

from fma_code import utils as fma_utils

N_FFT = 2048
HOP_LENGTH = 512
SAMPLING_RATE = 44100

AUDIO_PATH = os.path.join(os.getcwd(), 'fma_medium')


def load_audio_file(track_id: int) -> tuple[np.ndarray, float]:
    file_path = fma_utils.get_audio_path(AUDIO_PATH, track_id)
    return librosa.load(file_path, sr=SAMPLING_RATE, mono=True)


def trim_audio(audio: np.ndarray) -> np.ndarray:
    return librosa.effects.trim(audio, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]


def calculate_splits_count(audio: np.ndarray, sampling_rate: int = SAMPLING_RATE, split_duration: float = 3.0) -> int:
    total_duration = librosa.get_duration(y=audio, sr=sampling_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return total_duration // split_duration + 1


def split_audio(audio: np.ndarray, sampling_rate: int = SAMPLING_RATE, split_duration: float = 3.0) -> list[np.ndarray]:
    n_splits = calculate_splits_count(audio, sampling_rate, split_duration)
    return np.array_split(audio, n_splits)


def load_melspectrogram(audio: np.ndarray, sampling_rate: int = SAMPLING_RATE) -> np.ndarray:
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_fft=2048, hop_length=512)
    return librosa.power_to_db(mel_spectrogram)


def load_mfcc_spectrogram(audio: np.ndarray, sampling_rate: int = SAMPLING_RATE) -> np.ndarray:
    # TODO check implementation
    return librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=20)


@lru_cache(maxsize=512)
def generate_track_melspectrograms(track_id: int, split_duration: float = 3.0) -> Generator[np.ndarray, np.ndarray]:
    audio, sampling_rate = load_audio_file(track_id)
    trimmed_audio = trim_audio(audio)
    audio_splits = split_audio(trimmed_audio, sampling_rate, split_duration)
    for split in audio_splits:
        yield load_melspectrogram(split, sampling_rate)


@lru_cache(maxsize=512)
def generate_track_mfcc_spectrograms(track_id: int, split_duration: float = 3.0) -> Generator[np.ndarray, np.ndarray]:
    audio, sampling_rate = load_audio_file(track_id)
    trimmmed_audio = trim_audio(audio)
    audio_splits = split_audio(trimmmed_audio, sampling_rate, split_duration)
    for split in audio_splits:
        yield load_mfcc_spectrogram(split, sampling_rate)



class SpectrogramGenerator(keras.utils.Sequence):
    _N_FFT = N_FFT
    _HOP_LENGTH = HOP_LENGTH
    _SAMPLING_RATE = SAMPLING_RATE

    def __init__(
            self,
            track_ids: pd.Series,
            labels: pd.DataFrame,
            sample_duration: float = 3.0,
            batch_size: int = 128,
            spectrogram_dimensions: tuple[int, int] = (128, 130),
            shuffle: bool = True
        ) -> None:
        self._track_ids = track_ids
        self._labels = labels
        self._sample_duration = sample_duration
        self._batch_size = batch_size
        self._spectrogram_dimensions = spectrogram_dimensions
        self._shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self) -> int:
        return int(np.floor(len(self._track_ids) / self._batch_size))
    
    def on_epoch_end(self) -> None:
        self._indexes = np.array(self._track_ids.index)
        if self._shuffle:
            np.random.shuffle(self._indexes)
    
    def __getitem__(self, index: int) -> tuple[np.ndarray | np.ndarray]:
        indexes = self._indexes[index * self._batch_size: (index + 1) * self._batch_size]
        self._current_tracks_ids = self._track_ids.loc[indexes].index
        X = self._generate_X()
        y = self._generate_y()
        return X, y

    def _generate_X(self) -> np.ndarray:
        X = []
        for current_track_id in self._current_tracks_ids:
            try:
                current_track_spectrograms = generate_track_melspectrograms(current_track_id, self._sample_duration)
                for current_track_spectrogram in current_track_spectrograms:
                    if current_track_spectrogram.shape != self._spectrogram_dimensions:
                        current_track_spectrogram = np.resize(current_track_spectrogram, self._spectrogram_dimensions)
                    X.append(current_track_spectrogram)
            except FileNotFoundError:
                self._current_tracks_ids = self._current_tracks_ids.drop(current_track_id)
                self._track_ids.drop(current_track_id, inplace=True)
        return np.asarray(X)
    
    def _generate_y(self) -> np.ndarray:
        y = []
        for current_track_id in self._current_tracks_ids:
            y.extend(self._labels.loc[current_track_id])
        return np.array(y)


def create_model(labels_count: int, spectrogram_dimensions: tuple[int, int], channels: int) -> Sequential:
    model = Sequential([
        Input(shape=(*spectrogram_dimensions, channels, labels_count,)),
        Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='relu'),
        MaxPooling2D(),
        Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='relu'),
        MaxPooling2D(),
        Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(labels_count, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


def validate_labels_count(train_labels: pd.DataFrame, val_labels: pd.DataFrame, test_labels: pd.DataFrame):
    train_labels_count = train_labels.shape[1]
    val_labels_count = val_labels.shape[1]
    test_labels_count = test_labels.shape[1]
    assert train_labels_count == val_labels_count == test_labels_count, \
        f'Labels count is not equal: train={train_labels_count}, val={val_labels_count}, test={test_labels_count}'


def train_model(
        train_ids: list[int],
        val_ids: list[int],
        test_ids: list[int],
        train_labels: pd.DataFrame,
        val_labels: pd.DataFrame,
        test_labels: pd.DataFrame,
        sample_duration: float = 3.0,
        spectrogram_dimensions: tuple[int, int] = (128, 130),
        channels: int = 1,
        batch_size: int = 128,
        shuffle: bool = True) -> None:
    validate_labels_count(train_labels, val_labels, test_labels)

    model = create_model(train_labels.shape[1], spectrogram_dimensions, channels)

    # train_generator = SpectrogramGenerator(train_ids, train_labels, sample_duration, batch_size, spectrogram_dimensions, shuffle)
    # val_generator = SpectrogramGenerator(val_ids, val_labels, sample_duration, batch_size, spectrogram_dimensions, shuffle)
    # test_generator = SpectrogramGenerator(test_ids, test_labels, sample_duration, batch_size, spectrogram_dimensions, shuffle)

    # early_stopping = EarlyStopping(patience=7, restore_best_weights=True)
    # model.fit(
    #     train_generator,
    #     validation_data=val_generator,
    #     batch_size=batch_size,
    #     epochs=100,
    #     callbacks=[early_stopping],
    #     verbose=2
    # )

    # print(model.evaluate(test_generator))
