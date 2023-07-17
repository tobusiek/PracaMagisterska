import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

from audioread.exceptions import NoBackendError
import librosa
import numpy as np
import soundfile as sf

from fma_code import utils as fma_utils

N_FFT = 2048
HOP_LENGTH = 512
SAMPLING_RATE = 22050
SPLIT_DURATION = 3.0
REPOPULATE = True

AUDIO_PATH = os.path.join(os.getcwd(), 'fma_medium')
SPECTROGRAMS_PATH = os.path.join(os.getcwd(), 'pickled_spectrograms')


def load_audio_file(track_id: int) -> tuple[np.ndarray, int] | tuple[None, None]:
    file_path = fma_utils.get_audio_path(AUDIO_PATH, track_id)
    try:
        return librosa.load(file_path, sr=SAMPLING_RATE, mono=True)
    except (FileNotFoundError, sf.LibsndfileError, NoBackendError) as e:
        return None, None


def trim_audio(audio: np.ndarray) -> np.ndarray:
    return librosa.effects.trim(audio, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]


def calculate_splits_count(audio: np.ndarray, sampling_rate: int = SAMPLING_RATE, split_duration: float = SPLIT_DURATION) -> int:
    total_duration = librosa.get_duration(y=audio, sr=sampling_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return total_duration // split_duration + 1


def split_audio(audio: np.ndarray, sampling_rate: int = SAMPLING_RATE, split_duration: float = SPLIT_DURATION) -> list[np.ndarray]:
    n_splits = calculate_splits_count(audio, sampling_rate, split_duration)
    return np.array_split(audio, n_splits)


def load_melspectrogram(audio: np.ndarray, sampling_rate: int = SAMPLING_RATE) -> np.ndarray:
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_fft=2048, hop_length=512)
    return librosa.power_to_db(mel_spectrogram)


def generate_track_melspectrograms(
        audio: np.ndarray,
        sampling_rate: int = SAMPLING_RATE,
        split_duration: float = SPLIT_DURATION,
        max_shape: tuple[int, int] = (0, 0)
    ) -> np.ndarray | None:
    track_melspectrograms = []
    
    if not isinstance(audio, np.ndarray):
        return None
    
    trimmed_audio = trim_audio(audio)
    audio_splits = split_audio(trimmed_audio, sampling_rate, split_duration)
    
    for split in audio_splits:
        melspectrogram = load_melspectrogram(split, sampling_rate)
        if melspectrogram.shape > max_shape:
            max_shape = melspectrogram.shape
        track_melspectrograms.append(melspectrogram)
    
    for idx, melspectrogram in enumerate(track_melspectrograms):
        if melspectrogram.shape == max_shape: continue
        track_melspectrograms[idx] = np.resize(melspectrogram, max_shape)

    return np.array(track_melspectrograms, dtype=np.float16)


""" Not used, for later maybe?
def load_mfcc_spectrogram(audio: np.ndarray, sampling_rate: int = SAMPLING_RATE) -> np.ndarray:
    # TODO check implementation
    return librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=20)


def generate_track_mfcc_spectrograms(track_id: int, split_duration: float = 3.0) -> Generator[np.ndarray, np.ndarray]:
    audio, sampling_rate = load_audio_file(track_id)
    trimmmed_audio = trim_audio(audio)
    audio_splits = split_audio(trimmmed_audio, sampling_rate, split_duration)
    for split in audio_splits:
        yield load_mfcc_spectrogram(split, sampling_rate)
"""


def pickle_spectrograms_set(
        X: np.ndarray,
        y: np.ndarray,
        filename: str,
        compressed: bool = False,
        repopulate: bool = REPOPULATE
    ) -> None:
    if not os.path.exists(SPECTROGRAMS_PATH):
        os.mkdir(SPECTROGRAMS_PATH)

    set_filename = f'{filename}_{SPLIT_DURATION}s'
    if compressed:
        set_filename += '_compressed'
    
    set_path = os.path.join(SPECTROGRAMS_PATH, f'{set_filename}.npz')
    if os.path.exists(set_path) and not repopulate:
        print(f'File exists in: {set_path}. Repopulation set to false, returning.')
        return
    
    if compressed:
        np.savez_compressed(set_path, X=X, y=y)
    else:
        np.savez(set_path, X=X, y=y)
