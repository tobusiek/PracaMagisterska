import asyncio
from dataclasses import dataclass, field
import json
import logging
import logging.config
from pathlib import Path

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer, ConsumerStoppedError
from aiokafka.structs import ConsumerRecord
import tensorflow as tf

logging.config.fileConfig(Path('resources', 'logging.ini'), disable_existing_loggers=False)
logging.getLogger('aiokafka').setLevel(logging.ERROR)
logger = logging.getLogger('predictions')

result_sender: AIOKafkaProducer = None
request_receiver: AIOKafkaConsumer = None

# model = tf.keras.models.load_model('my_model.h5')


@dataclass(frozen=True)
class PredictionResult:
    request_id: str
    first_genre: str = field(default='first_genre')
    first_genre_result: float = field(default=0.6)
    second_genre: str = field(default='second_genre')
    second_genre_result: float = field(default=0.3)
    third_genre: str = field(default='third_genre')
    third_genre_result: float = field(default=0.1)


class PredictionResultEncoder(json.JSONEncoder):
    def default(self, prediction_result: PredictionResult):
        return prediction_result.__dict__


async def _create_request_receiver() -> AIOKafkaConsumer:
    '''Create AIOKafkaConsumer for receiving requests from producer.'''

    logger.debug('creating request_receiver...')
    return AIOKafkaConsumer(
        'requests_topic',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id='requests-group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_commit_interval_ms=1000
    )


async def _create_result_sender() -> AIOKafkaProducer:
    '''Create AIOKafkaProducer for sending prediction results to producer.'''

    logger.debug('creating result_sender...')
    return AIOKafkaProducer(
        bootstrap_servers=['localhost:9092'], 
        value_serializer=lambda x: json.dumps(x).encode('utf-8'),
    )


async def get_request_receiver() -> AIOKafkaConsumer:
    '''Get request_receiver (AIOKafkaConsumer), create if not instantiated.'''

    global request_receiver
    if request_receiver is None:
        request_receiver = _create_request_receiver()
        logger.debug('request_receiver created')
    return request_receiver


async def get_result_sender() -> AIOKafkaProducer:
    '''Get result_sender (AIOKafkaProducer), create if not instantiated.'''

    global result_sender
    if result_sender is None:
        result_sender = _create_result_sender()
        logger.debug('result_sender created')
    return result_sender


async def _start_result_sender() -> None:
    '''Start result_sender (AIOKafkaProducer).'''

    logger.debug('starting result_sender...')
    await result_sender.start()
    logger.debug('result_sender started')


async def _start_request_receiver() -> None:
    '''Start request_receiver (AIOKafkaConsumer).'''
    
    logger.debug('starting request_receiver...')
    await request_receiver.start()
    logger.debug('request_receiver started')
    

async def _start_kafka() -> None:
    '''Start both result_sender (AIOKafkaProducer) and request_receiver (AIOKafkaConsumer).'''
    
    logger.debug('starting kafka...')
    await asyncio.gather(
        _start_result_sender(),
        _start_request_receiver(),
    )
    logger.debug('kafka started')


async def initialize_kafka() -> None:
    '''Create and start both result_sender (AIOKafkaProducer) and request_receiver (AIOKafkaConsumer).'''
    
    global result_sender, request_receiver
    logger.debug('initializing kafka...')
    result_sender, request_receiver = await asyncio.gather(
        _create_result_sender(),
        _create_request_receiver(),
    )
    logger.debug('request_receiver and result_sender created')
    await _start_kafka()


async def _stop_result_sender() -> None:
    '''Stop result_sender (AIOKafkaProducer).'''

    logger.debug('stopping result_sender...')
    await result_sender.stop()


async def _stop_request_receiver() -> None:
    '''Stop request_receiver (AIOKafkaConsumer).'''
    
    logger.debug('stopping request_receiver...')
    await request_receiver.stop()


async def stop_kafka() -> None:
    '''Stop both result_sender (AIOKafkaProducer) and request_receiver (AIOKafkaConsumer).'''
    
    logger.debug('stopping kafka...')
    await asyncio.gather(
        _stop_result_sender(),
        _stop_request_receiver(),
    )
    logger.debug('kafka stopped successfully')


async def _send_prediction_result(request_id: str, prediction_result: str, prediction_result_encoder: PredictionResultEncoder) -> None:
    '''Send prediction result to server.'''
    
    response = {'request_id': request_id, 'predicted_result': prediction_result}
    response = PredictionResult(request_id)
    result_sender = await get_result_sender()
    await result_sender.send_and_wait('results_topic', prediction_result_encoder.encode(response))
    logger.debug(f'result sent to producer: {response}')


async def perform_prediction(message: ConsumerRecord, prediction_result_encoder: PredictionResultEncoder) -> None:
    '''Perform prediction on data received from server.'''
    
    received_data: dict[str, str | int] = message.value
    logger.debug(f'new data received: {received_data}')
    # Use Tensorflow model to make prediction
    # prediction = model.predict(data)
    # Include predicted result and original request ID in message payload
    # response = {'request_id': message['request_id'], 'predicted_result': prediction.tolist()}
    await _send_prediction_result(received_data['request_id'], received_data['data'], prediction_result_encoder)


async def run_consumer() -> None:
    '''Initialize Kafka and start consuming messages from server.'''

    await initialize_kafka()
    request_receiver = await get_request_receiver()
    preduction_result_encoder = PredictionResultEncoder()
    while True:
        try:
            message: ConsumerRecord = await request_receiver.getone()
            await perform_prediction(message, preduction_result_encoder)
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
