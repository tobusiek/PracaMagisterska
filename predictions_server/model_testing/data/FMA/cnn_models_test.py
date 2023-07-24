import os

import numpy as np
import pandas as pd
from tensorflow_addons.metrics import F1Score

from utils import DS_PATH

pd.options.display.max_rows = None
pd.options.display.max_columns = None
pd.options.display.max_colwidth = None

TRACKS_PATH = os.path.join(DS_PATH, 'tracks_filtered_small.csv')
tracks = pd.read_csv(TRACKS_PATH, index_col=0)

labels = pd.get_dummies(tracks['genre'], prefix='', prefix_sep='').columns
labels_count = len(labels)

del tracks

SPECTROGRAMS_PATH = os.path.join(os.getcwd(), 'pickled_spectrograms')
SPLIT_DURATION = 3.0

def get_set(
        set_name: str,
        spectrograms_path: str = SPECTROGRAMS_PATH,
        duration: float = SPLIT_DURATION
    ) -> np.ndarray:
    set_file_path = os.path.join(spectrograms_path, f'{set_name}_{duration}s.npz')
    return np.load(set_file_path, mmap_mode='r')

train_set = get_set('train_small')
x_train = train_set['X']
y_train = train_set['y']
del train_set

max_value = np.max(x_train)

input_shape = x_train[0].shape

from cnn_models import (
    compile_model, evaluate_saved_models, train_models,
    SpectrogramGenerator
)

BATCH_SIZE = 128

training_data_generator = SpectrogramGenerator(x_train, y_train, batch_size=BATCH_SIZE)
del x_train, y_train

val_set = get_set('validation_small')
x_val = val_set['X']
y_val = val_set['y']
del val_set

validation_data_generator = SpectrogramGenerator(x_val, y_val, batch_size=BATCH_SIZE)
del x_val, y_val

from keras import Model
from keras.models import Sequential
from keras.layers import (
    AveragePooling2D, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D,
    Rescaling, BatchNormalization, GlobalAveragePooling2D)
from keras.regularizers import L1L2, L1, L2

input = Input((*input_shape, 1))


def create_model1() -> Model:
    return compile_model(Sequential([
        input,
        Rescaling(1. / max_value),
        Conv2D(16, kernel_size=2, strides=2, activation='relu'),
        MaxPooling2D(),
        Conv2D(16, kernel_size=2, strides=2, activation='relu'),
        MaxPooling2D(),
        Conv2D(32, kernel_size=2, strides=2, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(labels_count, activation='softmax')
    ]))


def create_model2() -> Model:
    return compile_model(Sequential([
        input,
        Rescaling(1 / max_value),

        Conv2D(16, kernel_size=(3, 1), strides=2, activation='relu'),
        Conv2D(32, kernel_size=(3, 1), strides=2, activation='relu'),
        MaxPooling2D(),

        Conv2D(32, kernel_size=(3, 1), strides=2, activation='relu'),
        Conv2D(64, kernel_size=(3, 1), strides=2, activation='relu'),
        MaxPooling2D(),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.3),

        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(labels_count, activation='softmax'),
    ]))


def create_model3() -> Model:
    return compile_model(Sequential([
        input,
        Rescaling(1 / max_value),

        Conv2D(32, kernel_size=3, strides=2, kernel_regularizer=L1L2(l1=1e-3, l2=1e-4), activation='relu'),
        MaxPooling2D(),

        Conv2D(64, kernel_size=3, strides=2, kernel_regularizer=L1L2(l1=1e-3, l2=1e-4), activation='relu'),
        MaxPooling2D(),

        Conv2D(128, kernel_size=3, strides=2, kernel_regularizer=L1L2(l1=1e-3, l2=1e-4), activation='relu'),
        MaxPooling2D(),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),

        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(labels_count, activation='softmax'),
    ]))


def create_model4() -> Model:
    return compile_model(Sequential([
        input,
        Rescaling(1 / max_value),

        Conv2D(32, kernel_size=3, strides=2, kernel_regularizer=L1L2(l1=1e-3, l2=1e-4), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),

        Conv2D(64, kernel_size=3, strides=2, kernel_regularizer=L1L2(l1=1e-3, l2=1e-4), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),

        Conv2D(128, kernel_size=3, strides=2, kernel_regularizer=L1L2(l1=1e-3, l2=1e-4), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),

        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(labels_count, activation='softmax'),
    ]))


def create_model5() -> Model:
    return compile_model(Sequential([
        input,
        Rescaling(1 / max_value),

        Conv2D(64, kernel_size=3, strides=2, kernel_regularizer=L1L2(l1=1e-3, l2=1e-4), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),

        Conv2D(128, kernel_size=3, strides=2, kernel_regularizer=L1L2(l1=1e-3, l2=1e-4), activation='relu'),
        MaxPooling2D(),
        Dropout(0.5),

        Conv2D(256, kernel_size=3, strides=2, kernel_regularizer=L1L2(l1=1e-3, l2=1e-4), activation='relu'),
        GlobalAveragePooling2D(),

        Dense(1024, activation='relu'),
        Dropout(0.5),

        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(labels_count, activation='softmax'),
    ]))


def create_model6() -> Model: 
    return compile_model(Sequential([
        input,
        Rescaling(1 / max_value),

        Conv2D(64, kernel_size=3, strides=2, kernel_regularizer=L1L2(l1=1e-3, l2=1e-4), activation='relu'),
        MaxPooling2D(),

        Conv2D(128, kernel_size=3, strides=2, kernel_regularizer=L1L2(l1=1e-3, l2=1e-4), activation='relu'),
        MaxPooling2D(),

        Conv2D(256, kernel_size=3, strides=2, kernel_regularizer=L1L2(l1=1e-3, l2=1e-4), activation='relu'),
        MaxPooling2D(),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),

        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(labels_count, activation='softmax'),
    ]))


def create_model7() -> Model:
    return compile_model(Sequential([
        input,
        Rescaling(1 / max_value),

        Conv2D(64, kernel_size=3, strides=2, kernel_regularizer=L1L2(l1=1e-3, l2=1e-4), activation='relu'),
        MaxPooling2D(),

        Conv2D(128, kernel_size=3, strides=2, kernel_regularizer=L1L2(l1=1e-3, l2=1e-4), activation='relu'),
        MaxPooling2D(),

        Conv2D(256, kernel_size=3, strides=2, kernel_regularizer=L1L2(l1=1e-3, l2=1e-4), activation='relu'),
        MaxPooling2D(),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),

        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(labels_count, activation='softmax'),
    ]))


def create_model8() -> Model:
    return compile_model(Sequential([
        input,
        Rescaling(1 / max_value),

        Conv2D(64, kernel_size=2, strides=2, activation='relu'),
        MaxPooling2D(),

        Conv2D(128, kernel_size=2, strides=2, activation='relu'),
        MaxPooling2D(),

        Conv2D(256, kernel_size=2, strides=2, activation='relu'),
        MaxPooling2D(),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),

        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(labels_count, activation='softmax'),
    ]))


def create_model9() -> Model:
    return compile_model(Sequential([
        input,
        Rescaling(1 / max_value),

        Conv2D(64, kernel_size=2, strides=2, activation='relu'),
        MaxPooling2D(),

        Conv2D(128, kernel_size=2, strides=2, activation='relu'),
        MaxPooling2D(),

        Conv2D(256, kernel_size=2, strides=2, activation='relu'),
        MaxPooling2D(),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),

        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(labels_count, activation='softmax'),
    ]))


def create_model10() -> Model:
    return compile_model(Sequential([
        input,
        Rescaling(1 / max_value),

        Conv2D(64, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Conv2D(128, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Conv2D(256, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),

        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(labels_count, activation='softmax'),
    ]))


def create_model11() -> Model:
    return compile_model(Sequential([
        input,
        Rescaling(1 / max_value),

        Conv2D(64, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Conv2D(128, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Conv2D(256, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Conv2D(256, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),

        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(labels_count, activation='softmax'),
    ]))


def create_model12() -> Model:
    return compile_model(Sequential([
        input,
        Rescaling(1 / max_value),

        Conv2D(64, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Conv2D(128, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Conv2D(256, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Conv2D(256, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Conv2D(512, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),

        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(labels_count, activation='softmax'),
    ]))


def create_model13() -> Model:
    return compile_model(Sequential([
        input,
        Rescaling(1 / max_value),

        Conv2D(64, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Conv2D(128, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Conv2D(256, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),
        Dropout(0.5),

        Conv2D(256, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Conv2D(512, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Conv2D(512, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),

        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(labels_count, activation='softmax'),
    ]))


def create_model13_f1() -> Model:
    return compile_model(Sequential([
        input,
        Rescaling(1 / max_value),

        Conv2D(64, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Conv2D(128, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Conv2D(256, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),
        Dropout(0.5),

        Conv2D(256, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Conv2D(512, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Conv2D(512, kernel_size=2, strides=1, activation='relu'),
        MaxPooling2D(),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),

        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(labels_count, activation='softmax'),
    ]), F1Score(labels_count, average='micro'))


def create_model14_f1() -> Model:
    return compile_model(Sequential([
        input,
        Rescaling(1 / max_value),

        Conv2D(
            64, 
            kernel_size=2,
            strides=1, 
            activation='relu',),
        AveragePooling2D(),

        Conv2D(
            128,
            kernel_size=2,
            strides=1, 
            activation='relu',),
        AveragePooling2D(),

        Conv2D(
            256,
            kernel_size=2,
            strides=1, 
            activation='relu',),
        AveragePooling2D(),
        Dropout(0.5),

        Conv2D(
            256,
            kernel_size=2,
            strides=1, 
            activation='relu',),
        AveragePooling2D(),

        Conv2D(
            512,
            kernel_size=2,
            strides=1, 
            activation='relu',),
        AveragePooling2D(),

        Conv2D(
            512,
            kernel_size=2,
            strides=1, 
            activation='relu',),
        MaxPooling2D(),

        Flatten(),
        Dense(
            1024,
            activation='relu',),
        Dropout(0.5),

        Dense(
            512,
            activation='relu',),
        Dense(
            256,
            activation='relu',),
        Dense(
            128,
            activation='relu',),
        Dense(
            64,
            activation='relu',),
        Dense(labels_count, activation='softmax'),
    ]), F1Score(labels_count, average='micro'))


def create_model15_f1() -> Model:
    return compile_model(Sequential([
        input,
        Rescaling(1 / max_value),

        Conv2D(64, kernel_size=2, strides=1, activation='relu'),
        AveragePooling2D(),

        Conv2D(128, kernel_size=2, strides=1, activation='relu'),
        AveragePooling2D(),

        Conv2D(256, kernel_size=2, strides=1, activation='relu'),
        AveragePooling2D(),
        Dropout(0.5),

        Conv2D(256, kernel_size=2, strides=1, activation='relu'),
        AveragePooling2D(),

        Conv2D(512, kernel_size=2, strides=1, activation='relu'),
        AveragePooling2D(),

        Conv2D(512, kernel_size=2, strides=1, activation='relu'),
        GlobalAveragePooling2D(),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),

        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu',),
        Dense(64, activation='relu'),
        Dense(labels_count, activation='softmax'),
    ]), F1Score(labels_count, average='micro'))


def create_model15() -> Model:
    return compile_model(Sequential([
        input,
        Rescaling(1 / max_value),

        Conv2D(64, kernel_size=2, strides=1, activation='elu'),
        AveragePooling2D(),

        Conv2D(128, kernel_size=2, strides=1, activation='elu'),
        AveragePooling2D(),

        Conv2D(256, kernel_size=2, strides=1, activation='elu'),
        AveragePooling2D(),
        Dropout(0.5),

        Conv2D(256, kernel_size=2, strides=1, activation='elu'),
        AveragePooling2D(),

        Conv2D(512, kernel_size=2, strides=1, activation='elu'),
        AveragePooling2D(),

        Conv2D(512, kernel_size=2, strides=1, activation='elu'),
        GlobalAveragePooling2D(),

        Flatten(),
        Dense(1024, activation='elu'),
        Dropout(0.5),

        Dense(512, activation='elu'),
        Dense(256, activation='elu'),
        Dense(128, activation='elu',),
        Dense(64, activation='elu'),
        Dense(labels_count, activation='softmax'),
    ]))


models = {
    # 'model1': create_model1,
    # 'model2': create_model2,
    # 'model3': create_model3,
    # 'model4': create_model4,  # best 57%
    # 'model5': create_model5,
    # 'model6': create_model6,
    # 'model7': create_model7,
    # 'model8': create_model8,
    # 'model9': create_model9,  # best 57%
    # 'model10': create_model10,  # best 62.5%
    # 'model11': create_model11,  # best 64%
    # 'model12': create_model12,  # best 64.5%
    # 'model13': create_model13,  # best 67%
    # 'model13_f1': create_model13_f1,  # best 69%
    # 'model14_f1': create_model14_f1,  # best 68%
    # 'model15_f1': create_model15_f1,  # best 71%
    'model15': create_model15,  # 
}

models = train_models(
    models,
    training_data_generator,
    validation_data_generator=validation_data_generator,
    batch_size=BATCH_SIZE,
    epochs=150,
    model_checkpoint_monitor='accuracy',
    reduce_lr_monitor='accuracy'
)

del training_data_generator, validation_data_generator

test_set = get_set('test_small')
x_test = test_set['X']
y_test = test_set['y']
del test_set

test_data_generator = SpectrogramGenerator(x_test, y_test, batch_size=BATCH_SIZE, to_fit=True)
del x_test, y_test

evaluate_saved_models(list(models.keys()), test_data_generator, BATCH_SIZE)
