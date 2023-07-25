import numpy as np
from keras.models import Model, load_model

from data_models import PredictionResultModel
from predictions.audio_preprocessors.audio_features_preprocessor import AudioFeaturesPreprocessor
from predictions.models.base_prediction_model import BasePredictionModel
from tools.const_variables import MODEL_DIR, FMA_DATASET_INFO


class PredictionFeaturesModel(BasePredictionModel):
    '''Class for model to make predictions on audio features.'''

    _model_path = MODEL_DIR / 'simple_model.h5'
    _model: Model = load_model(_model_path)
    _audio_preprocessor = AudioFeaturesPreprocessor()
    _labels = FMA_DATASET_INFO['labels']
    _labels_decoded = {label_id: label for label, label_id in _labels.items()}
    _genres_to_take = 2
