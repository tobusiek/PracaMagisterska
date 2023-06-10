from json import load
from os import cpu_count
from pathlib import Path


MODEL_DIR = Path('model')
CORES_TO_USE = cpu_count() // 2
DATASET_AUDIO_FORMAT = '.au'
DATASET_INFO_PATH = MODEL_DIR / 'dataset_info.json'
with open(DATASET_INFO_PATH) as dataset_info:
    DATASET_INFO: dict[str, int | dict[str, float]] = load(dataset_info)
