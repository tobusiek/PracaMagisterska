import base64
from dataclasses import dataclass
from hashlib import sha3_224
import logging
import math
from pathlib import Path
from typing import AsyncGenerator, BinaryIO

from aiokafka.structs import ConsumerRecord
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from producer_setup import CHUNK_SIZE, MessageKey, get_result_receiver

logger = logging.getLogger('message_processor')


@dataclass
class FileChunk:
    '''Dataclass for storing chunks of audio files.'''

    chunk_number: int
    num_of_chunks: int
    chunk_data: bytes
    file_extension: str


def get_file_extension(filename: str) -> str:
    '''Get the file extension of uploaded file.'''

    return Path(filename).suffix


async def read_file(file: BinaryIO) -> bytes:
    return file.read()


async def chunkify_file(file_data: bytes, file_extension: str) -> AsyncGenerator[FileChunk, None]:
    '''Create chunks from audio file to assure they fit in kafka message and yield FileChunk.'''

    file_size = len(file_data)
    num_of_chunks = math.ceil(file_size / CHUNK_SIZE)
    logger.debug(f'splitting audio file to {num_of_chunks} chunks')
    for chunk_number in range(num_of_chunks):
        chunk_data_start_idx = chunk_number * CHUNK_SIZE
        chunk_data_stop_idx = (chunk_number + 1) * CHUNK_SIZE
        chunk_data = file_data[chunk_data_start_idx: chunk_data_stop_idx]
        if not chunk_data:
            return
        yield FileChunk(chunk_number, num_of_chunks, chunk_data, file_extension)


def _encode_file_chunk_with_base64(file_data_chunk: bytes) -> str:
    '''Encode file chunk with base64 encoding and return decoded string.'''

    return base64.b64encode(file_data_chunk).decode()


def checksum(file_data: bytes, request_id: str) -> str:
    '''Checksum on file_data.'''

    checksum_result = sha3_224(file_data).hexdigest()
    logger.debug(f'checksum for {request_id=}: {checksum_result}')
    return checksum_result


def create_message_with_file_chunk(request_id: str, file_chunk: FileChunk, file_checkum: str) -> dict[str, int | bytes | str]:
    '''Create a message with the given file chunk to be send to the prediction server.'''
    
    message = {
            MessageKey.REQUEST_ID.value: request_id,
            MessageKey.CHUNK_NUMBER.value: file_chunk.chunk_number,
            MessageKey.NUM_OF_CHUNKS.value: file_chunk.num_of_chunks,
            MessageKey.CHUNK_DATA.value: _encode_file_chunk_with_base64(file_chunk.chunk_data),
            MessageKey.FILE_EXTENSION.value: file_chunk.file_extension,
            MessageKey.CHECKSUM.value: file_checkum
        }
    logger.info(f'created new message with file chunk for {request_id=}: chunk {file_chunk.chunk_number} out of {file_chunk.num_of_chunks}')
    return message


async def receive_prediction_result(request_id: str) -> dict[str, str | float] | JSONResponse | None:
    '''Receive prediction result based on specified request id.'''

    result_receiver = await get_result_receiver()
    message: ConsumerRecord
    async for message in result_receiver:
        response: dict[str, str | float] = message.value
        response_id = response['request_id']
        if response_id == request_id:
            try:
                return response
            except ValidationError as e:
                return JSONResponse(content={'error': str(e)}, status_code=400)
