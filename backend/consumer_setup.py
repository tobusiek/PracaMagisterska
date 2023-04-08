import asyncio
import json
import logging

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

logger = logging.getLogger('consumer')

_request_receiver: AIOKafkaConsumer = None
_result_sender: AIOKafkaProducer = None


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

    global _request_receiver
    if _request_receiver is None:
        _request_receiver = _create_request_receiver()
        logger.debug('request_receiver created')
    return _request_receiver


async def get_result_sender() -> AIOKafkaProducer:
    '''Get result_sender (AIOKafkaProducer), create if not instantiated.'''

    global _result_sender
    if _result_sender is None:
        _result_sender = _create_result_sender()
        logger.debug('result_sender created')
    return _result_sender


async def _start_result_sender() -> None:
    '''Start result_sender (AIOKafkaProducer).'''

    logger.debug('starting result_sender...')
    await _result_sender.start()
    logger.debug('result_sender started')


async def _start_request_receiver() -> None:
    '''Start request_receiver (AIOKafkaConsumer).'''
    
    logger.debug('starting request_receiver...')
    await _request_receiver.start()
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
    
    global _result_sender, _request_receiver
    logger.debug('initializing kafka...')
    _result_sender, _request_receiver = await asyncio.gather(
        _create_result_sender(),
        _create_request_receiver(),
    )
    logger.debug('request_receiver and result_sender created')
    await _start_kafka()


async def _stop_result_sender() -> None:
    '''Stop result_sender (AIOKafkaProducer).'''

    logger.debug('stopping result_sender...')
    await _result_sender.stop()


async def _stop_request_receiver() -> None:
    '''Stop request_receiver (AIOKafkaConsumer).'''
    
    logger.debug('stopping request_receiver...')
    await _request_receiver.stop()


async def stop_kafka() -> None:
    '''Stop both result_sender (AIOKafkaProducer) and request_receiver (AIOKafkaConsumer).'''
    
    logger.debug('stopping kafka...')
    await asyncio.gather(
        _stop_result_sender(),
        _stop_request_receiver(),
    )
    logger.debug('kafka stopped successfully')
