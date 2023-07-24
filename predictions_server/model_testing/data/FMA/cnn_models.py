from typing import Callable

from keras.backend import clear_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras.utils
import numpy as np

from utils import create_model_path, save_model_history

BATCH_SIZE = 128


class SpectrogramGenerator(keras.utils.Sequence):
    def __init__(
            self,
            tracks_spectrograms: np.ndarray,
            labels: np.ndarray,
            sample_duration: float = 3.0,
            batch_size: int = BATCH_SIZE,
            shuffle: bool = True,
            to_fit: bool = True
        ) -> None:
        self._tracks_spectrograms = np.expand_dims(tracks_spectrograms, axis=-1)
        self._labels = labels
        self._sample_duration = sample_duration
        self._tracks_spectrograms_shape = tracks_spectrograms[0].shape
        self._batch_size = batch_size
        if not to_fit: shuffle = False
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


def compile_model(model: keras.Model, metric: str = 'accuracy') -> keras.Model:
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[metric])
    print(model.summary())
    return model


def train_models(
        models_creators: dict[str, Callable[[None], keras.Model]],
        training_data_generator: SpectrogramGenerator,
        validation_data_generator: SpectrogramGenerator = None,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
        epochs: int = 100,
        early_stopping_patience: int = 5,
        early_stopping_monitor: str = 'loss',
        model_checkpoint_monitor: str = 'accuracy',
        reduce_learning_rate: bool = True,
        reduce_lr_monitor: str = 'accuracy',
        reduce_lr_patience: int = 5,
        start_epoch: int = 10,
        verbose: int = 1
) -> dict[str, tuple[keras.Model, dict[str, float]]]:
    models_training_info: dict[str, tuple[keras.Model, dict[str, float]]] = {}
    for model_name, model_creator in models_creators.items():
        model_path = create_model_path(model_name)
        print('Model', model_name)
        model, model_history = train_model(
            model_creator(), training_data_generator, model_path,
            validation_data_generator, batch_size, shuffle, epochs,
            early_stopping_patience, early_stopping_monitor,
            model_checkpoint_monitor, reduce_learning_rate, reduce_lr_patience,
            reduce_lr_monitor, start_epoch, verbose
        )
        save_model_history(model_name, model_history)
        models_training_info[model_name] = model_history
        print()
    return models_training_info


def train_model(
        model: keras.Model,
        training_data_generator: SpectrogramGenerator,
        model_filepath: str,
        validation_data_generator: SpectrogramGenerator = None,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
        epochs: int = 100,
        early_stopping_patience: int = 5,
        early_stopping_monitor: str = 'loss',
        model_checkpoint_monitor: str = 'accuracy',
        reduce_learning_rate: bool = True,
        reduce_lr_patience: int = 5,
        reduce_lr_monitor: str = 'accuracy',
        start_epoch: int = 10,
        verbose: int = 1
    ) -> tuple[keras.Model, dict]:
    if validation_data_generator:
        early_stopping_monitor = 'val_' + early_stopping_monitor
        model_checkpoint_monitor = 'val_' + model_checkpoint_monitor
        reduce_lr_monitor = 'val_' + reduce_lr_monitor

    callbacks = [
        EarlyStopping(
            patience=early_stopping_patience, monitor=early_stopping_monitor,
            mode='min', restore_best_weights=True,
            start_from_epoch=start_epoch, verbose=verbose),
        ModelCheckpoint(
           model_filepath, monitor=model_checkpoint_monitor, mode='max',
            verbose=0, save_best_only=True)
    ]
        
    if reduce_learning_rate:
        callbacks.append(ReduceLROnPlateau(
            monitor=reduce_lr_monitor, mode='max', factor=0.5,
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

    clear_session()

    return model, model_fit.history


def evaluate_saved_models(
        models_names: list[str],
        test_data_generator: SpectrogramGenerator,
        batch_size: int = BATCH_SIZE
    ) -> None:
    for model_name in models_names:
        model_path = create_model_path(model_name)
        model = keras.models.load_model(model_path)
        evaluation = evaluate_model(model, test_data_generator, batch_size)
        print('Model', model_name, evaluation)


def evaluate_model(
        model: keras.Model,
        test_data_generator: SpectrogramGenerator,
        batch_size: int = BATCH_SIZE
    ) -> dict[str, float]:
    evaluation = model.evaluate(test_data_generator, batch_size=batch_size, return_dict=True)
    return evaluation


def model_prediction(
        model: keras.Model,
        test_data_generator: SpectrogramGenerator,
        batch_size: int = BATCH_SIZE
    ) -> np.ndarray:
    return model.predict(test_data_generator, batch_size=batch_size)
