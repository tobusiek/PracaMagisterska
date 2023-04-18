import asyncio
import base64
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
    await result_sender.send_and_wait('results_topic', response.make_dict())
    logger.debug(f'result sent to producer: {response}')


async def perform_prediction(message: ConsumerRecord) -> None:
    '''Perform prediction on data received from server.'''
    
    received_data: dict[str, str | int] = message.value
    logger.debug(f'new data received: {received_data}')
    await _send_prediction_result(received_data['request_id'], received_data['data'])


def _decode_file_chunk_with_base64(file_chunk: str) -> bytes:
    return base64.b64encode(file_chunk.encode('utf-8'))


async def _create_file_from_chunks(request_id: str) -> bytes | None:
    request = REQUESTS_BUFFER[request_id]
    if None in request:
        return
    file_data = b''.join(request)
    logger.debug(f'got whole file for {request_id}')
    return file_data


async def _perform_prediction_on_file(request_id: str, file_data: bytes, model: PredictionModel, file_extension: str):
    prediction_result = model.predict(request_id, file_data, file_extension)
    await _send_prediction_result(prediction_result)



async def process_messages(message: ConsumerRecord, model: PredictionModel) -> None:
    message_content = message.value
    file_chunk_request = FileChunkRequest(**message_content)
    request_id = file_chunk_request.request_id
    if request_id not in REQUESTS_BUFFER:
        REQUESTS_BUFFER[request_id] = [None] * file_chunk_request.num_of_chunks
    REQUESTS_BUFFER[request_id][file_chunk_request.chunk_number] = _decode_file_chunk_with_base64(file_chunk_request.chunk_data)
    file_data = await _create_file_from_chunks(request_id)
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
