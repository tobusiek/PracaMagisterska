import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

DS_PATH = os.path.join(os.getcwd(), 'fma_metadata')
MODELS_PATH = os.path.dirname(os.path.dirname(os.getcwd()))


def get_value_counts(df: pd.DataFrame, col_name: str | tuple[str, ...], min_occurrences: int = 0) -> dict[str | int | float, int]:
    value_counts = {}
    for value, occurrences in df[col_name].value_counts().items():
        if occurrences > min_occurrences:
            value_counts[value] = occurrences
    print(len(value_counts), 'values left')
    return pd.Series(value_counts)


def draw_pie(df: pd.DataFrame, col_name: str, min_occurrences: int = 0) -> None:
    value_counts = get_value_counts(df, col_name, min_occurrences)
    fig = px.pie(df[col_name], values=value_counts.values, names=value_counts.index)
    fig.update_traces(hoverinfo='label+percent', textinfo='percent')
    fig.show()


def draw_training_history(model_name: str, model_history: dict, metric: str = 'accuracy') -> None:
    train_acc = model_history[metric]
    val_acc = model_history[f'val_{metric}']
    train_loss = model_history['loss']
    val_loss = model_history['val_loss']
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(6, 6))

    metric = metric.capitalize()

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_acc, label=f'Training {metric}')
    plt.plot(epochs, val_acc, label=f'Validation {metric}')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.title(f'{model_name.capitalize()} Training and Validation {metric}')
    plt.legend(loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(model_name.capitalize() + ' Training and Validation Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(MODELS_PATH, '_'.join(model_name.split(' ')) + '.png'))


def draw_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> None:
    if len(y_true) > len(y_pred):
        length_difference = len(y_true) - len(y_pred)
        y_true = y_true[:-length_difference]
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display.plot(ax=ax)

    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)

    plt.ylabel('Predicted genre')
    plt.xlabel('True genre')
    plt.show()


def create_model_path(model_name: str) -> str:
    return os.path.join(MODELS_PATH, model_name + '.h5')


def save_model_history(model_name: str, models_history: dict[str, float]) -> None:
    models_history_path = os.path.join(MODELS_PATH, model_name + '.pickle')
    with open(models_history_path, 'wb+') as models_history_file:
        return pickle.dump(models_history, models_history_file)


def load_model_history(model_name: str) -> dict[str, float]:
    models_history_path = os.path.join(MODELS_PATH, model_name + '.pickle')
    with open(models_history_path, 'rb') as models_history_file:
        return pickle.load(models_history_file)
