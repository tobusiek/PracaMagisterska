import asyncio
import json
import logging
import logging.config
from pathlib import Path

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import tensorflow as tf

# Set up logging
logging.config.fileConfig(Path('resources', 'logging.ini'), disable_existing_loggers=False)
logging.getLogger('aiokafka').setLevel(logging.ERROR)
logger = logging.getLogger('debug')

# Load Tensorflow model
# model = tf.keras.models.load_model('my_model.h5')


async def create_request_receiver() -> AIOKafkaConsumer:
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


async def create_result_sender() -> AIOKafkaProducer:
    logger.debug('creating result_sender...')
    return AIOKafkaProducer(
        bootstrap_servers=['localhost:9092'], 
        value_serializer=lambda x: json.dumps(x).encode('utf-8'),
    )


async def start_kafka(result_sender: AIOKafkaProducer, request_receiver: AIOKafkaConsumer) -> None:
    logger.debug('starting kafka...')
    await asyncio.gather(
        result_sender.start(),
        request_receiver.start(),
    )
    logger.debug('kafka started!')


async def initialize_kafka() -> tuple[AIOKafkaProducer, AIOKafkaConsumer]:
    logger.debug('initializing kafka...')
    result_sender, request_receiver = await asyncio.gather(
        create_result_sender(),
        create_request_receiver(),
    )
    logger.debug('request_receiver and result_sender created')
    await start_kafka(result_sender, request_receiver)
    return result_sender, request_receiver


async def stop_result_sender(result_sender: AIOKafkaProducer) -> None:
    logger.debug('stopping result_sender...')
    await result_sender.stop()


async def stop_request_receiver(request_receiver: AIOKafkaConsumer) -> None:
    logger.debug('stopping request_receiver...')
    await request_receiver.stop()


async def stop_kafka(result_sender: AIOKafkaProducer, request_receiver: AIOKafkaConsumer) -> None:
    logger.debug('stopping kafka...')
    await asyncio.gather(
        stop_result_sender(result_sender),
        stop_request_receiver(request_receiver),
    )
    logger.debug('kafka stopped successfully')


# Define function to perform prediction and send response to Kafka output_topic
async def perform_prediction(message, result_sender: AIOKafkaProducer) -> None:
    # Extract the data to be predicted from the message payload
    data = message.value
    logger.debug(f'new data received: {data}, {type(message)}')
    # Use Tensorflow model to make prediction
    # prediction = model.predict(data)
    # Include predicted result and original request ID in message payload
    # response = {'request_id': message['request_id'], 'predicted_result': prediction.tolist()}
    response = {'request_id': data['request_id'], 'predicted_result': data['data']}
    # Send message to Kafka topic
    await result_sender.send_and_wait('results_topic', response)
    logger.debug(f'result sent to producer: {response}')


async def main() -> None: 
    result_sender, request_receiver = await initialize_kafka()
    while True:
        message = await request_receiver.getone()
        await perform_prediction(message, result_sender)
    logger.info('interrupted by user, quitting consumer')
    await stop_kafka(result_sender, request_receiver)


if __name__ == '__main__':
    logger.info('Consumer started')
    asyncio.run(main())
