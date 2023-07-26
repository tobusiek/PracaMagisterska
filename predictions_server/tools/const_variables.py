from json import load
from os import cpu_count, getcwd
from pathlib import Path

RESULTS_TOPIC = 'results_topic'

CORES_TO_USE = cpu_count() // 2

DATASETS_PATH = Path(getcwd(), 'model_testing', 'data')
FMA_OR_GTZAN = 'FMA'

GTZAN_DATASET_INFO_PATH = DATASETS_PATH / 'GTZAN' / 'dataset_info.json'
with open(GTZAN_DATASET_INFO_PATH) as gtzan_dataset_info:
    GTZAN_DATASET_INFO: dict = load(gtzan_dataset_info)

FMA_DATASET_INFO_PATH = DATASETS_PATH / 'FMA' /'dataset_info.json'
with open(FMA_DATASET_INFO_PATH) as fma_dataset_info:
    FMA_DATASET_INFO: dict = load(fma_dataset_info)

MODELS_PATH = Path(getcwd(), 'model_testing', 'models')
GTZAN_MODEL_PATH = MODELS_PATH / 'GTZAN' / 'simple_model.h5'
FMA_MODEL_PATH = MODELS_PATH / 'FMA' / 'best_model.h5'

TEMP_FILES_PATH = Path(getcwd(), 'temp_files')
