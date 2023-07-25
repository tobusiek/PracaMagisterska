from abc import ABC, abstractmethod

from librosa.effects import trim
import numpy as np

from predictions.temp_file_creator import TempFileCreator


class BaseAudioPreprocessor(ABC):
    """Base class for audio preprocessing."""

    _temp_file_creator = TempFileCreator()

    @abstractmethod
    def preprocess_audio(self, request_id: str, file_data: bytes, filex_extension: str) -> np.ndarray:
        raise NotImplemented("Called preprocess_audio on an abstract class")

    @staticmethod
    def _trim_audio(audio: np.ndarray) -> np.ndarray:
        """Trim audio to get rid of silent parts."""

        return trim(audio)[0]

    @abstractmethod
    def _split_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        raise NotImplemented("Called split_audio on an abstract class")

    @abstractmethod
    def _trim_split(self, split: np.ndarray) -> np.ndarray:
        raise NotImplemented("Called trim_split on an abstract class")
