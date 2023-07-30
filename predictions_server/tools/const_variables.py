from json import load
from os import cpu_count, getcwd
from pathlib import Path

RESULTS_TOPIC = 'results_topic'

CORES_TO_USE = cpu_count() // 2

FMA_OR_GTZAN = 'GTZAN'


MODELS_PATH = Path(getcwd(), 'predictions', 'models')

GTZAN_DATASET_INFO_PATH = MODELS_PATH / 'gtzan' / 'dataset_info.json'
with open(GTZAN_DATASET_INFO_PATH) as gtzan_dataset_info:
    GTZAN_DATASET_INFO: dict = load(gtzan_dataset_info)

FMA_DATASET_INFO_PATH = MODELS_PATH / 'fma' / 'dataset_info.json'
with open(FMA_DATASET_INFO_PATH) as fma_dataset_info:
    FMA_DATASET_INFO: dict = load(fma_dataset_info)

MIN_AUDIO_DURATION: float = FMA_DATASET_INFO['split_duration'] if FMA_OR_GTZAN == 'FMA' else GTZAN_DATASET_INFO['split_duration']

MINUTE = 60
N_MINUTES = 10
MAX_AUDIO_DURATION = MINUTE * N_MINUTES

GTZAN_MODEL_PATH = MODELS_PATH / 'gtzan' / 'simple_model.h5'
FMA_MODEL_PATH = MODELS_PATH / 'fma' / 'best_model.h5'

TEMP_FILES_PATH = Path(getcwd(), 'temp_files')
