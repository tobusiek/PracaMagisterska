from json import load
from os import cpu_count
from pathlib import Path

FMA_OR_GTZAN = 'FMA'
MODEL_DIR = Path('model_testing', 'data', FMA_OR_GTZAN, 'models')
CORES_TO_USE = cpu_count() // 2
FMA_DATASET_INFO_PATH = Path('model_testing', 'data', FMA_OR_GTZAN, 'dataset_info.json')
with open(FMA_DATASET_INFO_PATH) as fma_dataset_info:
    FMA_DATASET_INFO: dict = load(fma_dataset_info)
