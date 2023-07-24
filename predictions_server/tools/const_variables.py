from json import load
from os import cpu_count
from pathlib import Path


MODEL_DIR = Path('model_testing', 'data', 'FMA', 'models')
CORES_TO_USE = cpu_count() // 2
DATASET_INFO_PATH = Path('model_testing', 'data', 'FMA', 'dataset_info.json')
with open(DATASET_INFO_PATH) as dataset_info:
    DATASET_INFO: dict = load(dataset_info)
