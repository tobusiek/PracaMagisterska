import json
import asyncio
import logging
import logging

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
logger = logging.getLogger('producer')


async def _create_request_sender() -> AIOKafkaProducer:
    '''Create AIOKafkaProducer for sending requests to consumer.'''

    logger.debug('creating request_sender...')
    return AIOKafkaProducer(
        bootstrap_servers=['localhost:9092'], 
        value_serializer=lambda x: json.dumps(x).encode('utf-8'),
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
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_commit_interval_ms=1000
    )


async def get_request_sender() -> AIOKafkaProducer:
    '''Get request_sender (AIOKafkaProducer), create if not instantiated.'''
    
    global request_sender
    if request_sender is None:
        request_sender = await _create_request_sender()
        logger.debug('request_sender created: {request_sender}')
    return request_sender


async def get_result_receiver() -> AIOKafkaConsumer:
    '''Get result_receiver (AIOKafkaConsumer), create if not instantiated.'''
    
    global result_receiver
    if result_receiver is None:
        result_receiver = await _create_result_receiver()
        logger.debug('result_receiver created: {result_receiver}')
    return result_receiver


async def _start_request_sender() -> None:
    '''Start request_sender (AIOKafkaProducer).'''

    logger.debug('starting request_sender...')
    await request_sender.start()
    logger.debug('request_sender started')


async def _start_result_receiver() -> None:
    '''Start result_receiver (AIOKafkaConsumer).'''

    logger.debug('starting result_receiver...')
    await result_receiver.start()
    logger.debug('result_receiver started')


async def _start_kafka() -> None:
    '''Start both request_sender (AIOKafkaProducer) and result_receiver (AIOKafkaConsumer).'''

    logger.debug('starting kafka...')
    await asyncio.gather(
        _start_request_sender(),
        _start_result_receiver(),
    )
    logger.debug('kafka started!')


async def initialize_kafka() -> None:
    '''Create and start both request_sender (AIOKafkaProducer) and result_receiver (AIOKafkaConsumer).'''

    global result_receiver, request_sender
    logger.debug('initializing kafka...')
    request_sender, result_receiver = await asyncio.gather(
        _create_request_sender(),
        _create_result_receiver(),
    )
    logger.debug('result_receiver and request_sender created')
    await _start_kafka()


async def _stop_request_sender() -> None:
    '''Stop request_sender (AIOKafkaProducer).'''

    logger.debug('stopping request_sender...')
    await request_sender.stop()


async def _stop_result_receiver() -> None:
    '''Stop result_receiver (AIOKafkaConsumer).'''
    
    logger.debug('stopping result receiver...')
    await result_receiver.stop()


async def stop_kafka() -> None:
    '''Stop both request_sender (AIOKafkaProducer) and result_receiver (AIOKafkaConsumer).'''

    logger.debug('stopping kafka...')
    await asyncio.gather(
        _stop_request_sender(),
        _stop_result_receiver(),
    )
    logger.debug('kafka stopped successfully')
