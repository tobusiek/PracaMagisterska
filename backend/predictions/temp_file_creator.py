import logging
from pathlib import Path

from pydub import AudioSegment

from tools.const_variables import MODEL_DIR

logger = logging.getLogger('file_creator')


class TempFileCreator:
    '''Class for creating temporary files.'''

    _files_temp_path = MODEL_DIR / 'temp_files'

    def _create_files_temp_path(self) -> None:
        '''Create directory for temporary files if not exists.'''

        self._files_temp_path.mkdir(exist_ok=True)
        logger.info('created directory for temporary files')

    def _create_temp_wav_file(self, request_id: str, temp_filename: str) -> Path:
        '''Create temporary .wav file from other extensions.'''
        
        temp_filename_wav = request_id + '.wav'
        audio_wav_path = self._files_temp_path / temp_filename_wav
        audio_segment = AudioSegment.from_file(self._files_temp_path / temp_filename)
        audio_segment.export(audio_wav_path, format='wav')
        logger.info(f'created temporary .wav file for {request_id=}')
        return audio_wav_path

    def create_temp_file(self, request_id: str, file_data: bytes, file_extension: str) -> Path:
        '''Create temporary audio file. If extension is not .wav, create second file in wav format.'''
        
        self._create_files_temp_path()
        temp_file_name = request_id + file_extension
        temp_file_path = Path(self._files_temp_path / temp_file_name)
        with open(temp_file_path, 'wb+') as temp_file:
            temp_file.write(file_data)
        logger.info(f'created temporary {file_extension} file for {request_id=}')
        if file_extension != '.wav':
            temp_file_path = self._create_temp_wav_file(request_id, temp_file_name)
        return temp_file_path
    
    def delete_temp_file(self, request_id: str) -> None:
        '''Delete temporary audio file(s). If extension was not .wav, delete both files.'''
        
        for dir_el in self._files_temp_path.iterdir():
            if dir_el.is_dir():
                continue
            if request_id in dir_el.name:
                dir_el.unlink()
        logger.info(f'deleted temporary audio file(s) for {request_id=}')
