import logging
from abc import ABC, abstractmethod
from pathlib import Path

from audioread.exceptions import NoBackendError
from librosa import load, get_duration
from librosa.effects import trim
import numpy as np
import soundfile as sf

from predictions.audio_preprocessors.audio_exceptions import CorruptedAudioFileException
from predictions.temp_file_creator import TempFileCreator

logger = logging.getLogger('preprocessor')


class BaseAudioPreprocessor(ABC):
    """Base class for audio preprocessing."""

    _temp_file_creator: TempFileCreator
    _split_duration: float
    _sampling_rate: int
    _n_fft: int
    _hop_length: int

    @abstractmethod
    def preprocess_audio(self, request_id: str, file_data: bytes, filex_extension: str) -> np.ndarray:
        raise NotImplemented("Called preprocess_audio on an abstract class")

    def _get_audio(self, request_id: str, file_data: bytes, file_extension: str) -> np.ndarray:
        """Load audio from file. Create temporary file from bytes and load it with librosa."""

        logger.info(f'preprocessing audio for {request_id=}...')
        temp_file_path = self._temp_file_creator.create_temp_file(request_id, file_data, file_extension)
        audio, sampling_rate = self._load_audio_file(temp_file_path)
        self._temp_file_creator.delete_temp_file(request_id)
        if not sampling_rate:
            logger.error(f'corrupted audio file for {request_id=}')
            raise CorruptedAudioFileException(f'corrupted audio file for {request_id=}')
        self._validate_audio_duration(request_id, audio)
        return audio

    def _load_audio_file(self, file_path: Path) -> tuple[np.ndarray, float] | tuple[None, None]:
        """Try loading audio file given the path."""

        try:
            return load(file_path, sr=self._sampling_rate, mono=True)
        except (FileNotFoundError, sf.LibsndfileError, NoBackendError):
            return None, None

    @staticmethod
    def _trim_audio(audio: np.ndarray) -> np.ndarray:
        """Trim audio to get rid of silent parts."""

        return trim(audio)[0]

    def _get_audio_duration(self, audio: np.ndarray) -> float:
        """Get duration of audio in seconds."""

        return get_duration(
            y=audio,
            sr=self._sampling_rate,
            n_fft=self._n_fft,
            hop_length=self._hop_length
        )

    @abstractmethod
    def _validate_audio_duration(self, request_id: str, audio: np.ndarray) -> None:
        raise NotImplemented("Called _validate_audio_duration on an abstract class")

    @abstractmethod
    def _split_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        raise NotImplemented("Called split_audio on an abstract class")

    @abstractmethod
    def _trim_split(self, split: np.ndarray) -> np.ndarray:
        raise NotImplemented("Called trim_split on an abstract class")
