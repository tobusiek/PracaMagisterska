from heapq import nlargest

import numpy as np
import tensorflow as tf

from data_models import PredictionResultModel
from .audio_preprocessor import AudioPreprocessor
from tools.const_variables import MODEL_DIR, DATASET_INFO


class PredictionModel:
    '''Class for model to make predictions on audio file.'''

    _model_path = MODEL_DIR / 'simple_model.h5'
    _model: tf.keras.models.Model = tf.keras.models.load_model(_model_path)
    _audio_preprocessor = AudioPreprocessor()
    _labels = DATASET_INFO['labels']
    _labels_decoded = {label_id: label for label, label_id in _labels.items()}
    _genres_to_take = 3

    def _get_n_predictions_with_largest_means(self, prediction: np.ndarray[np.ndarray[np.float32]]) -> list[tuple[str, float]]:
        '''Get n largest means from model prediction.'''
        
        predicted_genres_means = [np.mean(prediction_for_genre) for prediction_for_genre in prediction.T]
        largest_means_of_predicted_genres = nlargest(self._genres_to_take, predicted_genres_means)
        indices_of_n_largest_means = [np.where(predicted_genres_means == nth_max)[0][0] for nth_max in largest_means_of_predicted_genres]
        return [(self._labels_decoded[idx], round(100 * nth_largest_mean, 2))
                for idx, nth_largest_mean in zip(indices_of_n_largest_means, largest_means_of_predicted_genres)]

    def _create_prediction_result_model(self, request_id: str, prediction: np.ndarray[np.ndarray[np.float32]]) -> PredictionResultModel:
        '''Create a model prediction result for n largest means from model prediction.'''

        n_predictions_with_largest_means = self._get_n_predictions_with_largest_means(prediction)
        prediction_with_first_largest_mean = n_predictions_with_largest_means[0]
        prediction_with_second_largest_mean = n_predictions_with_largest_means[1]
        prediction_with_third_largest_mean = n_predictions_with_largest_means[2]
        return PredictionResultModel(
            request_id=request_id,
            first_genre=prediction_with_first_largest_mean[0],
            first_genre_result=prediction_with_first_largest_mean[1],
            second_genre=prediction_with_second_largest_mean[0],
            second_genre_result=prediction_with_second_largest_mean[1],
            third_genre=prediction_with_third_largest_mean[0],
            third_genre_result=prediction_with_third_largest_mean[1])


    def predict(self, request_id: str, file_data: bytes, file_extension: str) -> PredictionResultModel:
        '''Make a prediction on audio file.'''

        audio_df = self._audio_preprocessor.preprocess_audio(request_id, file_data, file_extension)
        prediction = self._model.predict(audio_df)
        return self._create_prediction_result_model(request_id, prediction)
