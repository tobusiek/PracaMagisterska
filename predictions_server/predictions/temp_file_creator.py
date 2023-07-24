import logging
from pathlib import Path

from pydub import AudioSegment

from tools.const_variables import DATASET_INFO, MODEL_DIR

logger = logging.getLogger('file_creator')


class TempFileCreator:
    '''Class for creating temporary files.'''

    _temp_files_path = MODEL_DIR / 'temp_files'
    _dataset_audio_format: str = DATASET_INFO['audio_format']

    def _create_files_temp_path(self) -> None:
        '''Create directory for temporary files if not exists.'''

        self._temp_files_path.mkdir(exist_ok=True)
        logger.info('created directory for temporary files')

    def _create_temp_file_with_dataset_audio_format(self, request_id: str, temp_file_path: Path) -> Path:
        '''Create temporary file with dataset's audio format from other extensions.'''
        
        temp_filename_dataset_audio_format = request_id + self._dataset_audio_format
        proper_format_audio_path = self._temp_files_path / temp_filename_dataset_audio_format
        audio_segment: AudioSegment = AudioSegment.from_file(temp_file_path)
        audio_segment.export(proper_format_audio_path, format=self._dataset_audio_format.replace('.', ''))
        logger.info(f"created temporary file with dataset's audio format for {request_id=}")
        return proper_format_audio_path

    def create_temp_file(self, request_id: str, file_data: bytes, file_extension: str) -> Path:
        '''Create temporary audio file. If extension is not the same as dataset's audio format, create second file with proper format.'''
        
        self._create_files_temp_path()
        if not file_extension.startswith('.'):
            file_extension = '.' + file_extension
        temp_file_name = request_id + file_extension
        temp_file_path = Path(self._temp_files_path / temp_file_name)
        with open(temp_file_path, 'wb+') as temp_file:
            temp_file.write(file_data)
        logger.info(f'created temporary {file_extension} file for {request_id=}')
        if file_extension != self._dataset_audio_format:
            temp_file_path = self._create_temp_file_with_dataset_audio_format(request_id, temp_file_path)
        return temp_file_path
    
    def delete_temp_file(self, request_id: str) -> None:
        '''Delete temporary audio file(s). If extension was not the same as dataset's audio format, delete both files.'''
        
        for dir_element in self._temp_files_path.iterdir():
            if dir_element.is_dir():
                continue
            if request_id in dir_element.name:
                dir_element.unlink()
        logger.info(f'deleted temporary audio file(s) for {request_id=}')
