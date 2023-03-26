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


def create_consumer() -> AIOKafkaConsumer:
    return AIOKafkaConsumer(
        'requests_topic',
        bootstrap_servers='localhost:9092',
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id='requests-group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_commit_interval_ms=1000
    )


def create_producer() -> AIOKafkaProducer:
    return AIOKafkaProducer(
        bootstrap_servers='localhost:9092', 
        value_serializer=lambda x: json.dumps(x).encode('utf-8'),
    )


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
    request_receiver = create_consumer()
    # Initialize Kafka producer
    result_sender = create_producer()
    try:
        # Start Kafka producer
        await result_sender.start()
        # Start Kafka consumer
        await request_receiver.start()
        # Start consumer loop to process incoming messages
        while True:
            message = await request_receiver.getone()
            await perform_prediction(message, result_sender)
    except KeyboardInterrupt:
        logger.info('Consumer stopped')
    finally:
        await request_receiver.stop()
        await result_sender.stop()
        quit()


if __name__ == '__main__':
    logger.info('Consumer started')
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
    loop.close()
