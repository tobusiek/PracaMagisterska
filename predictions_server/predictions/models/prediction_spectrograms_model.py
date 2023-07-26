from keras.models import Model, load_model

from predictions.audio_preprocessors.audio_spectrograms_preprocessor import AudioSpectrogramsPreprocessor
from predictions.models.base_prediction_model import BasePredictionModel
from tools.const_variables import FMA_MODEL_PATH, FMA_DATASET_INFO


class PredictionSpectrogramsModel(BasePredictionModel):
    """Class for model to make predictions on audio spectrograms."""

    def __init__(self):
        model: Model = load_model(FMA_MODEL_PATH)
        labels: list[str] = FMA_DATASET_INFO['labels']
        super().__init__(model, AudioSpectrogramsPreprocessor(), labels)
