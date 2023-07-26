from keras.models import Model, load_model

from predictions.audio_preprocessors.audio_features_preprocessor import AudioFeaturesPreprocessor
from predictions.models.base_prediction_model import BasePredictionModel
from tools.const_variables import GTZAN_MODEL_PATH, GTZAN_DATASET_INFO


class PredictionFeaturesModel(BasePredictionModel):
    """Class for model to make predictions on audio features."""

    def __init__(self):
        model: Model = load_model(GTZAN_MODEL_PATH)
        labels: list[str] = GTZAN_DATASET_INFO['labels']
        super().__init__(model, AudioFeaturesPreprocessor(), labels)
