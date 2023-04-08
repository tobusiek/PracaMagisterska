import asyncio
import logging.config
from pathlib import Path

from aiokafka import ConsumerStoppedError
from aiokafka.structs import ConsumerRecord
import tensorflow as tf

from consumer_setup import initialize_kafka, stop_kafka, get_request_receiver, get_result_sender
from prediction_result_model import PredictionResultModel


logging.config.fileConfig(Path('resources', 'logging.ini'), disable_existing_loggers=False)
logging.getLogger('aiokafka').setLevel(logging.ERROR)
logger = logging.getLogger('predictions')

# model = tf.keras.models.load_model('my_model.h5')


def _create_prediction_result_message(
        request_id: str,
        first_genre: str = 'first_genre',
        first_genre_result: float = 0.6,
        second_genre: str = 'second_genre',
        second_genre_result: float = 0.3,
        third_genre: str = 'third_genre',
        third_genre_result: float = 0.1
    ) -> PredictionResultModel:
    '''Create prediction result message.'''

    return PredictionResultModel(request_id,
        first_genre, first_genre_result,
        second_genre, second_genre_result,
        third_genre, third_genre_result)


async def _send_prediction_result(request_id: str, prediction_result: str) -> None:
    '''Send prediction result to server.'''
    
    response = _create_prediction_result_message(request_id)
    result_sender = await get_result_sender()
    await result_sender.send_and_wait('results_topic', response.make_dict())
    logger.debug(f'result sent to producer: {response}')


async def perform_prediction(message: ConsumerRecord) -> None:
    '''Perform prediction on data received from server.'''
    
    received_data: dict[str, str | int] = message.value
    logger.debug(f'new data received: {received_data}')
    await _send_prediction_result(received_data['request_id'], received_data['data'])


async def run_consumer() -> None:
    '''Initialize Kafka and start consuming messages from server.'''

    await initialize_kafka()
    request_receiver = await get_request_receiver()
    while True:
        try:
            message: ConsumerRecord = await request_receiver.getone()
            await perform_prediction(message)
        except ConsumerStoppedError:
            return


async def main():
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
