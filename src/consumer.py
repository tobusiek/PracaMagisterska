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


async def create_consumer() -> AIOKafkaConsumer:
    logger.debug('creating consumer')
    return AIOKafkaConsumer(
        'requests_topic',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id='requests-group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_commit_interval_ms=1000
    )


async def create_producer() -> AIOKafkaProducer:
    logger.debug('creating producer...')
    return AIOKafkaProducer(
        bootstrap_servers=['localhost:9092'], 
        value_serializer=lambda x: json.dumps(x).encode('utf-8'),
    )


async def stop_kafka(result_sender: AIOKafkaProducer, request_receiver: AIOKafkaConsumer) -> None:
    await result_sender.stop()
    logger.debug('stopped producer')
    await request_receiver.stop()
    logger.debug('stopped consumer')


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
    # Initialize Kafka consumer
    request_receiver = await create_consumer()
    logger.debug('request_receiver created')
    # Initialize Kafka producer
    result_sender = await create_producer()
    logger.debug('result_sender created')
    try:
        # Start Kafka producer
        await result_sender.start()
        logger.debug('result_sender started')
        # Start Kafka consumer
        await request_receiver.start()
        logger.debug('request_receiver started')
        # Start consumer loop to process incoming messages
        while True:
            message = await request_receiver.getone()
            await perform_prediction(message, result_sender)
    except KeyboardInterrupt:
        logger.info('Consumer stopped')
    finally:
        await stop_kafka(result_sender, request_receiver)
        quit()


if __name__ == '__main__':
    logger.info('Consumer started')
    asyncio.run(main())
