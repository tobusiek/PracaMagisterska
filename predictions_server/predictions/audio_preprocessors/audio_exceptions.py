class AudioTooLongException(Exception):
    """Error class for too long audio. Raised when audio duration > MAX_AUDIO_DURATION."""
    ...


class AudioTooShortException(Exception):
    """Error class for too short audio. Raised when audio duration < split duration."""
    ...


class CorruptedAudioFileException(Exception):
    """Error class for corrupted audio. Raised when audio could not be loaded by librosa."""
    ...
