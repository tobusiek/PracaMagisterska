import logging
import multiprocessing as mp
import warnings

from librosa import power_to_db
from librosa.effects import trim
from librosa.feature import melspectrogram
import numpy as np

from predictions.audio_preprocessors.audio_exceptions import AudioTooShortException, AudioTooLongException
from predictions.audio_preprocessors.base_audio_preprocessor import BaseAudioPreprocessor
from predictions.temp_file_creator import TempFileCreator
from tools.const_variables import FMA_DATASET_INFO, CORES_TO_USE, MAX_AUDIO_DURATION

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

logger = logging.getLogger('preprocessor')


class AudioSpectrogramsPreprocessor(BaseAudioPreprocessor):
    """Class for audio spectrograms preprocessing."""

    _temp_file_creator = TempFileCreator(FMA_DATASET_INFO['audio_format'])
    _split_duration: float = FMA_DATASET_INFO['split_duration']
    _sampling_rate: int = FMA_DATASET_INFO['sampling_rate']
    _n_fft: int = FMA_DATASET_INFO['n_fft']
    _hop_length: int = FMA_DATASET_INFO['hop_length']
    _spectrogram_shape: tuple[int, int] = tuple(FMA_DATASET_INFO['spectrogram_shape'])

    def preprocess_audio(self, request_id: str, file_data: bytes, file_extension: str) -> np.ndarray:
        """Create temporary file from bytes, load it with librosa, trim it, make spectrograms and delete temporary file."""

        audio = self._get_audio(request_id, file_data, file_extension)
        return self._generate_track_melspectrograms(audio)

    def _validate_audio_duration(self, request_id: str, audio: np.ndarray) -> None:
        duration = self._get_audio_duration(audio)
        if duration < self._split_duration:
            logger.warning(f'{request_id=} audio too short')
            raise AudioTooShortException(f'{request_id=} audio duration to short')
        if duration > MAX_AUDIO_DURATION:
            logger.warning(f'{request_id=} audio too long')
            raise AudioTooLongException(f'{request_id=} audio duration to long')

    def _generate_track_melspectrograms(self, audio: np.ndarray) -> np.ndarray | None:
        track_melspectrograms = []
        spectrogram_shape = self._spectrogram_shape

        if not isinstance(audio, np.ndarray):
            return None

        trimmed_audio = self._trim_audio(audio)
        audio_splits = self._split_audio(trimmed_audio)

        if not audio_splits:
            return None

        with mp.Pool(CORES_TO_USE) as pool:
            for split in audio_splits:
                trimmed_split = self._trim_split(split)
                melspectrogram = pool.apply(self._load_melspectrogram, args=(trimmed_split,))
                if melspectrogram.shape > spectrogram_shape:
                    spectrogram_shape = melspectrogram.shape
                track_melspectrograms.append(melspectrogram)

        for idx, melspectrogram in enumerate(track_melspectrograms):
            if melspectrogram.shape == spectrogram_shape:
                continue
            track_melspectrograms[idx] = np.resize(melspectrogram, spectrogram_shape)

        return np.array(track_melspectrograms, dtype=np.float16)

    def _trim_audio(self, audio: np.ndarray) -> np.ndarray:
        return trim(audio, frame_length=self._n_fft, hop_length=self._hop_length)[0]

    def _split_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        n_splits = self._calculate_splits_count(audio)
        if n_splits <= 0:
            return []
        return np.array_split(audio, n_splits)

    def _calculate_splits_count(self, audio: np.ndarray) -> float:
        total_duration = self._get_audio_duration(audio)
        return total_duration // self._split_duration

    def _trim_split(self, split: np.ndarray) -> np.ndarray:
        return split[:int(self._sampling_rate * self._split_duration)]

    def _load_melspectrogram(self, audio: np.ndarray) -> np.ndarray:
        mel_spectrogram = melspectrogram(
            y=audio,
            sr=self._sampling_rate,
            n_fft=self._n_fft,
            hop_length=self._hop_length)
        return power_to_db(mel_spectrogram)
