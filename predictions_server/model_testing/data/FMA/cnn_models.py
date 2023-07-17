from keras.backend import clear_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras.utils
from numba import cuda
import numpy as np

from utils import create_model_path


class SpectrogramGenerator(keras.utils.Sequence):
    def __init__(
            self,
            tracks_spectrograms: np.ndarray,
            labels: np.ndarray,
            sample_duration: float = 3.0,
            batch_size: int = 128,
            shuffle: bool = True,
            to_fit: bool = True
        ) -> None:
        self._tracks_spectrograms = np.expand_dims(tracks_spectrograms, axis=-1)
        self._labels = labels
        self._sample_duration = sample_duration
        self._tracks_spectrograms_shape = tracks_spectrograms[0].shape
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._to_fit = to_fit
        self.on_epoch_end()
    
    def __len__(self) -> int:
        return int(np.floor(len(self._tracks_spectrograms) / self._batch_size))
    
    def on_epoch_end(self) -> None:
        self._indexes = np.arange(len(self._tracks_spectrograms))
        if self._shuffle:
            np.random.shuffle(self._indexes)
    
    def __getitem__(self, index: int) -> np.ndarray | tuple[np.ndarray | np.ndarray]:
        current_spectrograms_idxs = self._indexes[
            index * self._batch_size: (index + 1) * self._batch_size]
        
        X = self._generate_X(current_spectrograms_idxs)
        
        if self._to_fit:
            y = self._generate_Y(current_spectrograms_idxs)
            return X, y
        
        return X

    def _generate_X(self, current_spectrograms_idxs: np.ndarray) -> np.ndarray:
        X = np.empty(
            (self._batch_size, *self._tracks_spectrograms_shape, 1), dtype=np.float16)

        for idx, current_spectrogram_idx in enumerate(current_spectrograms_idxs):
            X[idx,] = self._tracks_spectrograms[current_spectrogram_idx]
        
        return X
    
    def _generate_Y(self, current_labels_idxs: np.ndarray) -> np.ndarray:
        y = np.empty((self._batch_size, *self._labels[0].shape), dtype=np.uint8)

        for idx, current_label_idxs in enumerate(current_labels_idxs):
            y[idx,] = self._labels[current_label_idxs]
        
        return y


def reinit_gpu():
    clear_session()
    cuda.select_device(0)
    cuda.close()


def compile_model(model: keras.Model) -> keras.Model:
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def train_models(
        models_with_names: dict[str, keras.Model],
        training_data_generator: SpectrogramGenerator,
        metric: str = 'accuracy',
        validation_data_generator: SpectrogramGenerator = None,
        batch_size: int = 128,
        shuffle: bool = True,
        epochs: int = 100,
        early_stopping_patience: int = 5,
        reduce_learning_rate: bool = True,
        reduce_lr_patience: int = 5,
        start_epoch: int = 10,
        verbose: int = 1
) -> dict[str, tuple[keras.Model, dict[str, float]]]:
    models_training_info: dict[str, tuple[keras.Model, dict[str, float]]] = {}
    for model_name, model in models_with_names.items():
        model_path = create_model_path(model_name)
        print('Model', model_name)
        model, model_history = train_model(
            model, training_data_generator, model_path, metric,
            validation_data_generator, batch_size, shuffle, epochs,
            early_stopping_patience, reduce_learning_rate, reduce_lr_patience,
            start_epoch, verbose
        )
        models_training_info[model_name] = model, model_history
        print()
    return models_training_info


def train_model(
        model: keras.Model,
        training_data_generator: SpectrogramGenerator,
        model_filepath: str,
        metric: str = 'accuracy',
        validation_data_generator: SpectrogramGenerator = None,
        batch_size: int = 128,
        shuffle: bool = True,
        epochs: int = 100,
        early_stopping_patience: int = 5,
        reduce_learning_rate: bool = True,
        reduce_lr_patience: int = 5,
        start_epoch: int = 10,
        verbose: int = 1
    ) -> tuple[keras.Model, dict]:
    callbacks = None

    if validation_data_generator:
        callbacks = [
            EarlyStopping(
                patience=early_stopping_patience, monitor='val_loss', mode='min',
                restore_best_weights=True, start_from_epoch=start_epoch, verbose=verbose),
            ModelCheckpoint(
                model_filepath, monitor=f'val{metric}', mode='max', verbose=0,
                save_best_only=True)
        ]
        
        if reduce_learning_rate:
            callbacks.append(ReduceLROnPlateau(
                monitor=f'val_{metric}', mode='max', factor=0.5,
                patience=reduce_lr_patience, verbose=verbose, min_lr=1e-6)
            )

    model_fit = model.fit(
        training_data_generator,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=validation_data_generator,
        shuffle=shuffle,
        callbacks=callbacks
    )

    reinit_gpu()

    return model, model_fit.history


def evaluate_model(
        model: keras.Model,
        test_data_generator: SpectrogramGenerator,
        batch_size: int = 128
    ) -> dict[str, float]:
    evaluation = model.evaluate(test_data_generator, batch_size=batch_size, return_dict=True)
    return evaluation


def model_prediction(
        model: keras.Model,
        x_test: np.ndarray,
        y_test: np.ndarray,
        sample_duration: float = 3.0,
        batch_size: int = 128
    ) -> np.ndarray:
    test_data_generator = SpectrogramGenerator(
        x_test, y_test, sample_duration, batch_size, shuffle=False, to_fit=True)
    return model.predict(test_data_generator, batch_size=batch_size)
