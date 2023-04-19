import base64
from collections.abc import Generator
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
from typing import BinaryIO

from aiokafka.structs import ConsumerRecord
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from producer_setup import CHUNK_SIZE, MessageKey, get_result_receiver

logger = logging.getLogger('fastapi')


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


async def chunkify_file(file: BinaryIO, file_extension: str) -> Generator[FileChunk]:
    '''Create chunks from audio file to assure they fit in kafka message and yield FileChunk.'''

    file_size = os.fstat(file.fileno()).st_size
    num_of_chunks = math.ceil(file_size / CHUNK_SIZE)
    file.seek(0)
    for chunk_number in range(num_of_chunks):
        chunk_data = file.read(CHUNK_SIZE)
        if not chunk_data:
            break
        yield FileChunk(chunk_number, num_of_chunks, chunk_data, file_extension)


def _encode_file_chunk_with_base64(file_chunk: bytes) -> str:
    '''Encode file chunk with base64 encoding and return decoded string.'''

    return base64.b64encode(file_chunk).decode()


def create_message_with_file_chunk(request_id: str, file_chunk: FileChunk) -> dict[MessageKey, int | bytes]:
    '''Create a message with the given file chunk to be send to the prediction server.'''
    
    message = {
            MessageKey.REQUEST_ID.value: request_id,
            MessageKey.CHUNK_NUMBER.value: file_chunk.chunk_number,
            MessageKey.NUM_OF_CHUNKS.value: file_chunk.num_of_chunks,
            MessageKey.CHUNK_DATA.value: _encode_file_chunk_with_base64(file_chunk.chunk_data),
            MessageKey.FILE_EXTENSION.value: file_chunk.file_extension,
        }
    logger.debug(f'created new message with file chunk for {request_id=}: chunk {file_chunk.chunk_number} out of {file_chunk.num_of_chunks}')
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
