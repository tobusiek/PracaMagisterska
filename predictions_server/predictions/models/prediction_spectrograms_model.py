from collections import namedtuple
from heapq import nlargest
import logging

import numpy as np
from keras.models import Model, load_model

from predictions.audio_preprocessors.audio_spectrograms_preprocessor import AudioSpectrogramsPreprocessor
from data_models import PredictionResultModel
from tools.const_variables import MODEL_DIR, FMA_DATASET_INFO

logger = logging.getLogger('predictions')

GenresPrediction = namedtuple('GenresPrediction', ('genre', 'genre_result'))


class PredictionSpectrogramsModel:
    '''Class for model to make predictions on audio spectrograms.'''

    _model_path = MODEL_DIR / 'best_model.h5'
    _model: Model = load_model(_model_path)
    _audio_preprocessor = AudioSpectrogramsPreprocessor()
    _labels: list[str] = FMA_DATASET_INFO['labels']
    _genres_to_take = 2

    def _get_n_predictions_with_largest_means(self, prediction: np.ndarray[np.ndarray[np.float32]]) -> tuple[GenresPrediction, ...]:
        '''Get n largest means from model prediction.'''
        # TODO: refactor, cause it's ugly
        
        predicted_genres_means = [np.mean(prediction_for_genre) for prediction_for_genre in prediction.T]
        largest_means_of_predicted_genres = nlargest(self._genres_to_take, predicted_genres_means)
        indices_of_n_largest_means = [np.where(predicted_genres_means == nth_max)[0][0] for nth_max in largest_means_of_predicted_genres]
        logger.debug(f'largest means: {largest_means_of_predicted_genres}, indices for genres with largest means: {indices_of_n_largest_means}')
        return tuple(GenresPrediction(genre=self._labels[idx], genre_result=round(100 * nth_largest_mean, 2))
                for idx, nth_largest_mean in zip(indices_of_n_largest_means, largest_means_of_predicted_genres))

    def _create_prediction_result_model(self, prediction: np.ndarray[np.ndarray[np.float32]]) -> PredictionResultModel:
        '''Create a model prediction result for n largest means from model prediction.'''

        n_predictions_with_largest_means = self._get_n_predictions_with_largest_means(prediction)
        first_highest_result, second_highest_result = n_predictions_with_largest_means
        prediction_result = PredictionResultModel(
            first_genre=first_highest_result.genre,
            first_genre_result=first_highest_result.genre_result,
            second_genre=second_highest_result.genre,
            second_genre_result=second_highest_result.genre_result)
        return prediction_result

    def predict(self, request_id: str, file_data: bytes, file_extension: str) -> PredictionResultModel:
        '''Make a prediction on audio file.'''

        logger.debug(f'making prediction for {request_id=}...')
        processed_audio = self._audio_preprocessor.preprocess_audio(request_id, file_data, file_extension)
        prediction = self._model.predict(processed_audio)
        prediction_result = self._create_prediction_result_model(prediction)
        logger.info(f'prediction result for {request_id=}: {prediction_result}')
        return prediction_result