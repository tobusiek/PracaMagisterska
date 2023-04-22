from enum import Enum
import json
import asyncio
import logging

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

logger = logging.getLogger('producer')

_request_sender: AIOKafkaProducer = None
_result_receiver: AIOKafkaConsumer = None

CHUNK_SIZE = 1024 * 1024


class MessageKey(Enum):
    REQUEST_ID = 'request_id'
    CHUNK_NUMBER = 'chunk_number'
    NUM_OF_CHUNKS = 'num_of_chunks'
    CHUNK_DATA = 'chunk_data'
    FILE_EXTENSION = 'file_extension'


async def _create_request_sender() -> AIOKafkaProducer:
    '''Create AIOKafkaProducer for sending requests to consumer.'''

    logger.debug('creating request_sender...')
    return AIOKafkaProducer(
        bootstrap_servers=['localhost:9092'], 
        value_serializer=lambda message: json.dumps(message).encode('utf-8'),
        max_request_size=2 * CHUNK_SIZE,
    )


async def _create_result_receiver() -> AIOKafkaConsumer:
    '''Create AIOKafkaConsumer for receiving prediction results from consumer.'''

    logger.debug('creating result_receiver...')
    return AIOKafkaConsumer(
        'results_topic',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id='results-group',
        value_deserializer=lambda message: json.loads(message.decode('utf-8')),
        auto_commit_interval_ms=1_000
    )


async def get_request_sender() -> AIOKafkaProducer:
    '''Get request_sender (AIOKafkaProducer), create if not instantiated.'''
    
    global _request_sender
    if _request_sender is None:
        _request_sender = await _create_request_sender()
        logger.info('request_sender not present, new instance created')
    return _request_sender


async def get_result_receiver() -> AIOKafkaConsumer:
    '''Get result_receiver (AIOKafkaConsumer), create if not instantiated.'''
    
    global _result_receiver
    if _result_receiver is None:
        _result_receiver = await _create_result_receiver()
        logger.info('result_receiver not present, new instance created')
    return _result_receiver


async def _start_request_sender() -> None:
    '''Start request_sender (AIOKafkaProducer).'''

    logger.debug('starting request_sender...')
    await _request_sender.start()
    logger.info('request_sender started')


async def _start_result_receiver() -> None:
    '''Start result_receiver (AIOKafkaConsumer).'''

    logger.debug('starting result_receiver...')
    await _result_receiver.start()
    logger.info('result_receiver started')


async def _start_kafka() -> None:
    '''Start both request_sender (AIOKafkaProducer) and result_receiver (AIOKafkaConsumer).'''

    logger.debug('starting kafka...')
    await asyncio.gather(
        _start_request_sender(),
        _start_result_receiver(),
    )
    logger.info('kafka started!')


async def initialize_kafka() -> None:
    '''Create and start both request_sender (AIOKafkaProducer) and result_receiver (AIOKafkaConsumer).'''

    global _result_receiver, _request_sender
    logger.debug('initializing kafka...')
    _request_sender, _result_receiver = await asyncio.gather(
        _create_request_sender(),
        _create_result_receiver(),
    )
    logger.debug('result_receiver and request_sender created')
    await _start_kafka()


async def _stop_request_sender() -> None:
    '''Stop request_sender (AIOKafkaProducer).'''

    logger.debug('stopping request_sender...')
    await _request_sender.stop()


async def _stop_result_receiver() -> None:
    '''Stop result_receiver (AIOKafkaConsumer).'''
    
    logger.debug('stopping result receiver...')
    await _result_receiver.stop()


async def stop_kafka() -> None:
    '''Stop both request_sender (AIOKafkaProducer) and result_receiver (AIOKafkaConsumer).'''

    logger.debug('stopping kafka...')
    await asyncio.gather(
        _stop_request_sender(),
        _stop_result_receiver(),
    )
    logger.info('kafka stopped successfully')
