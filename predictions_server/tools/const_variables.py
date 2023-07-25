from json import load
from os import cpu_count
from pathlib import Path

CORES_TO_USE = cpu_count() // 2

GTZAN_DATASET_INFO_PATH = Path('model_testing', 'data', 'GTZAN', 'models', 'simple_model.h5')

FMA_MODEL_PATH = Path('model_testing', 'data', 'FMA', 'models', 'best_model.h5')
FMA_DATASET_INFO_PATH = Path('model_testing', 'data', 'FMA', 'dataset_info.json')
with open(FMA_DATASET_INFO_PATH) as fma_dataset_info:
    FMA_DATASET_INFO: dict = load(fma_dataset_info)
