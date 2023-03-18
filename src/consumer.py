import asyncio
import json
import logging
import logging.config
import os

from kafka import KafkaProducer, KafkaConsumer
from kafka.consumer.fetcher import ConsumerRecord
import tensorflow as tf

# Set up logging
logging.config.fileConfig(os.path.join('resources', 'logging.ini'), disable_existing_loggers=False)
logging.getLogger('kafka').setLevel(logging.ERROR)
logger = logging.getLogger('debug')

# Initialize Kafka producer
result_sender = KafkaProducer(
    bootstrap_servers=['localhost:9092'], 
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# Initialize Kafka consumer
request_receiver = KafkaConsumer(
    'requests_topic',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='requests-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
)

# Load Tensorflow model
# model = tf.keras.models.load_model('my_model.h5')


# Define function to perform prediction and send response to Kafka output_topic
def perform_prediction(message: ConsumerRecord) -> None:
    # Extract the data to be predicted from the message payload
    data = message.value
    logger.debug(f'new data received: {data}')

    # Use Tensorflow model to make prediction
    # prediction = model.predict(data)

    # Include predicted result and original request ID in message payload
    # response = {'request_id': message['request_id'], 'predicted_result': prediction.tolist()}
    response = {'request_id': data['request_id'], 'predicted_result': data['data']}

    # Send message to Kafka topic
    result_sender.send('results_topic', response)
    logger.debug(f'result sent to producer: {response}')


def main() -> None:
    # Start consumer loop to process incoming messages
    for message in request_receiver:
        perform_prediction(message)


if __name__ == '__main__':
    logger.info('Consumer started')
    main()
