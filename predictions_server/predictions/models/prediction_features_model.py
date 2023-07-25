from keras.models import Model, load_model

from predictions.audio_preprocessors.audio_features_preprocessor import AudioFeaturesPreprocessor
from predictions.models.base_prediction_model import BasePredictionModel
from tools.const_variables import MODEL_DIR, GTZAN_DATASET_INFO


class PredictionFeaturesModel(BasePredictionModel):
    '''Class for model to make predictions on audio features.'''
    _model_path = MODEL_DIR / 'simple_model.h5'
    _labels = GTZAN_DATASET_INFO['labels']

    def __init__(self):
        model: Model = load_model(self._model_path)

        super().__init__(model, AudioFeaturesPreprocessor(), self._labels)
