from collections import namedtuple
from heapq import nlargest
import logging

import numpy as np
from keras.models import Model, load_model

from data_models import PredictionResultModel
from .audio_features_preprocessor import AudioFeaturesPreprocessor
from tools.const_variables import MODEL_DIR, DATASET_INFO

logger = logging.getLogger('predictions')

GenresPrediction = namedtuple('GenresPrediction', ('genre', 'genre_result'))


class PredictionFeaturesModel:
    '''Class for model to make predictions on audio features.'''

    _model_path = MODEL_DIR / 'simple_model.h5'
    _model: Model = load_model(_model_path)
    _audio_preprocessor = AudioFeaturesPreprocessor()
    _labels = DATASET_INFO['labels']
    _labels_decoded = {label_id: label for label, label_id in _labels.items()}
    _genres_to_take = 3

    def _get_n_predictions_with_largest_means(self, prediction: np.ndarray[np.ndarray[np.float32]]) -> tuple[GenresPrediction, ...]:
        '''Get n largest means from model prediction.'''
        
        predicted_genres_means = [np.mean(prediction_for_genre) for prediction_for_genre in prediction.T]
        largest_means_of_predicted_genres = nlargest(self._genres_to_take, predicted_genres_means)
        indices_of_n_largest_means = [np.where(predicted_genres_means == nth_max)[0][0] for nth_max in largest_means_of_predicted_genres]
        logger.debug(f'largest means: {largest_means_of_predicted_genres}, indices for genres with largest means: {indices_of_n_largest_means}')
        return tuple(GenresPrediction(genre=self._labels_decoded[idx], genre_result=round(100 * nth_largest_mean, 2))
                for idx, nth_largest_mean in zip(indices_of_n_largest_means, largest_means_of_predicted_genres))

    def _create_prediction_result_model(self, request_id: str, prediction: np.ndarray[np.ndarray[np.float32]]) -> PredictionResultModel:
        '''Create a model prediction result for n largest means from model prediction.'''

        n_predictions_with_largest_means = self._get_n_predictions_with_largest_means(prediction)
        first_highest_result, second_highest_result, third_highest_result = n_predictions_with_largest_means
        prediction_result = PredictionResultModel(
            request_id=request_id,
            first_genre=first_highest_result.genre,
            first_genre_result=first_highest_result.genre_result,
            second_genre=second_highest_result.genre,
            second_genre_result=second_highest_result.genre_result,
            third_genre=third_highest_result.genre,
            third_genre_result=third_highest_result.genre_result)
        logger.info(f'prediction result for {request_id=}: {prediction_result}')
        return prediction_result


    def predict(self, request_id: str, file_data: bytes, file_extension: str) -> PredictionResultModel:
        '''Make a prediction on audio file.'''

        logger.debug(f'making prediction for {request_id=}...')
        processed_audio = self._audio_preprocessor.preprocess_audio(request_id, file_data, file_extension)
        prediction = self._model.predict(processed_audio)
        return self._create_prediction_result_model(request_id, prediction)
