import asyncio
import base64
from hashlib import sha3_224
import logging.config
from pathlib import Path

from aiokafka import ConsumerStoppedError
from aiokafka.structs import ConsumerRecord

from consumer_setup import initialize_kafka, stop_kafka, get_request_receiver, get_result_sender
from data_models import PredictionResultModel, FileChunkRequest
from predictions.prediction_model import PredictionModel


logging.config.fileConfig(Path('resources', 'logging.ini'), disable_existing_loggers=False)
logging.getLogger('aiokafka').setLevel(logging.ERROR)
logger = logging.getLogger('predictions')

REQUESTS_BUFFER: dict[str, list[bytes]] = {}



def _create_prediction_result_message(prediction_result: PredictionResultModel) -> dict[str, str | float]:
    '''Create prediction result message.'''

    return prediction_result.make_dict()


async def _send_prediction_result(prediction_result: PredictionResultModel) -> None:
    '''Send prediction result to server.'''
    
    response = _create_prediction_result_message(prediction_result)
    result_sender = await get_result_sender()
    await result_sender.send_and_wait('results_topic', response)
    logger.info(f'result sent to producer: {response}')


def _decode_file_chunk_with_base64(file_chunk: str) -> bytes:
    '''Decode file chunk received from producer (str) with base64 encoding.'''

    return base64.b64decode(file_chunk.encode())


def _checksum(file_data: bytes) -> str:
    return sha3_224(file_data).hexdigest()


def _compare_checksum(file_data: bytes, original_checksum: str) -> bool:
    return original_checksum == _checksum(file_data)


def _create_file_from_chunks(request_id: str, original_checksum: str) -> bytes | None:
    '''Create file from chunks if every chunk for sent file present in requests buffer.'''

    request = REQUESTS_BUFFER[request_id]
    if None in request:
        return
    file_data = b''.join(request)
    logger.debug(f'got whole file for {request_id}')
    if _compare_checksum(file_data, original_checksum):
        logger.debug(f'checksum for {request_id=} correct')
        return file_data
    raise BytesWarning('checksums differ')


async def _perform_prediction_on_file(request_id: str, file_data: bytes, model: PredictionModel, file_extension: str) -> None:
    '''Perform prediction on received file and send the results back to producer.'''
    
    prediction_result = model.predict(request_id, file_data, file_extension)
    await _send_prediction_result(prediction_result)


async def process_messages(message: ConsumerRecord, model: PredictionModel) -> None:
    '''Process messages received from producer, by putting them in requests buffer.
       If the whole file is received, make a prediction and send results back to producer.'''

    message_content = message.value
    file_chunk_request = FileChunkRequest(**message_content)
    request_id = file_chunk_request.request_id
    num_of_chunks = file_chunk_request.num_of_chunks
    if request_id not in REQUESTS_BUFFER:
        REQUESTS_BUFFER[request_id] = [None] * num_of_chunks
    chunk_number = file_chunk_request.chunk_number
    REQUESTS_BUFFER[request_id][chunk_number] = _decode_file_chunk_with_base64(file_chunk_request.chunk_data)
    file_checksum = file_chunk_request.file_checksum
    logger.info(f'new message for {request_id=}: {chunk_number=} out of {num_of_chunks}, {file_checksum}')
    try:
        file_data = _create_file_from_chunks(request_id, file_checksum)
    except BytesWarning as e:
        logger.error(f'error for{request_id=}: {e}')
    else:
        if file_data:
            await _perform_prediction_on_file(request_id, file_data, model, file_chunk_request.file_extension)


async def run_consumer() -> None:
    '''Initialize Kafka and start consuming messages from server.'''

    await initialize_kafka()
    request_receiver = await get_request_receiver()
    model = PredictionModel()
    while True:
        try:
            message: ConsumerRecord = await request_receiver.getone()
            await process_messages(message, model)
        except ConsumerStoppedError:
            return


async def main() -> None:
    await run_consumer()


if __name__ == '__main__':
    logger.info('starting consumer...')
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        if loop.is_closed():
            loop = asyncio.new_event_loop()
        loop.run_until_complete(stop_kafka())
