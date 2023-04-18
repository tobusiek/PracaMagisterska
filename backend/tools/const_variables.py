from json import load
from pathlib import Path


MODEL_DIR = Path('model')
DATASET_INFO_PATH = MODEL_DIR / 'dataset_info.json'
with open(DATASET_INFO_PATH) as dataset_info:
    DATASET_INFO: dict[str, int | dict[str, float]] = load(dataset_info)
