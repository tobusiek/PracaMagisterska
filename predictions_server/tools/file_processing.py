import base64
from hashlib import sha3_224
import logging

from data_models import FileChunkRequest

logger = logging.getLogger('predictions')

REQUESTS_BUFFER: dict[str, list[bytes]] = {}


class ChecksumDifferenceException(BytesWarning):
    ...


def create_file_from_chunks(request_id: str, original_checksum: str) -> bytes | None:
    """Create file from chunks if every chunk for sent file present in requests buffer."""

    request = REQUESTS_BUFFER[request_id]
    if None in request:
        return
    file_data = b''.join(request)
    logger.debug(f'got whole file for {request_id}')
    if _valid_checksum(file_data, original_checksum):
        logger.debug(f'checksum for {request_id=} correct')
        return file_data
    raise ChecksumDifferenceException(f'for {request_id=}')


def fill_buffer(request_id: str, file_chunk_request: FileChunkRequest) -> None:
    """Fill requests buffer with received chunk of file requested for prediction."""

    chunk_number = file_chunk_request.chunk_number
    num_of_chunks = file_chunk_request.num_of_chunks
    logger.info(f'new message for {request_id=}: chunk {chunk_number + 1} out of {num_of_chunks}')
    if request_id not in REQUESTS_BUFFER:
        REQUESTS_BUFFER[request_id] = [None] * num_of_chunks
    REQUESTS_BUFFER[request_id][chunk_number] = _decode_file_chunk_with_base64(file_chunk_request.chunk_data)


def remove_request_from_buffer(request_id: str) -> None:
    if request_id not in REQUESTS_BUFFER:
        return
    REQUESTS_BUFFER.pop(request_id, None)

def _decode_file_chunk_with_base64(file_chunk: str) -> bytes:
    """Decode file chunk received from producer (str) with base64 encoding."""

    return base64.b64decode(file_chunk.encode())


def _calculate_checksum(file_data: bytes) -> str:
    """Checksum on file_data."""

    return sha3_224(file_data).hexdigest()


def _valid_checksum(file_data: bytes, original_checksum: str) -> bool:
    """Compare original checksum with checksum after concatenation of message."""

    checksum_result = _calculate_checksum(file_data)
    logger.debug(f'original checksum={original_checksum}, checksum after concatenating={checksum_result}')
    return original_checksum == checksum_result
