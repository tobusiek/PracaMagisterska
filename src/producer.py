import asyncio
import json
import logging
import logging.config
from pathlib import Path
import uuid

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.structs import ConsumerRecord
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi import Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from uvicorn import Config, Server

logging.config.fileConfig(Path('resources', 'logging.ini'), disable_existing_loggers=False)
logging.getLogger('aiokafka').setLevel(logging.ERROR)
logger = logging.getLogger('debug')

app = FastAPI(debug=True)
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

request_sender: AIOKafkaProducer = None
result_receiver: AIOKafkaConsumer = None


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


async def start_server() -> None:
    '''Start FastAPI server.'''

    loop = asyncio.get_event_loop()
    config = Config(app, loop=loop, log_level='debug')
    server = Server(config)
    logger.debug('starting server...')
    await server.serve()
    logger.debug('server stopped')


def _create_message(request_id: str, data: str | int) -> dict[str, str | int]:
    '''Create message to be sent to the consumer.'''

    message = {'request_id': request_id, 'data': data}
    logger.debug(f'created new message: {message}')
    return message


async def _send_request_to_consumer(message: dict[str, str | int]) -> None:
    '''Send prediction request to consumer.'''

    request_sender = await get_request_sender()
    await request_sender.send_and_wait('requests_topic', message)
    await request_sender.flush()  # TODO try removing
    logger.debug(f'request sent to consumer: {message}')


async def receive_result(request_id: str) -> dict[str, str | int]:
    '''Receive message with prediction result from consumer. Returns prediction result for specific request.'''

    result_receiver = await get_result_receiver()
    message: ConsumerRecord
    async for message in result_receiver:
        response: dict[str, str] = message.value
        response_id = response['request_id']
        if response_id == request_id:
            try:
                predicted_result = response['predicted_result']
                result = {'predicted_result': predicted_result}
                return result
            except ValidationError as e:
                return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get('/', response_class=HTMLResponse)
async def root(request: Request):
    '''Endpoint for root, redirects to /prediction.'''

    return RedirectResponse(url='/predict', headers=request.headers)


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    '''Endpoint for favicon.ico.'''

    return FileResponse(Path('static', 'images', 'favicon.svg'))


@app.get('/predict', response_class=HTMLResponse)
async def get_predict(request: Request):
    '''Endpoint for making prediction requests.'''

    return templates.TemplateResponse('input-form.html', {'request': request})


@app.post('/predict')
async def post_predict(data_input: str | int = Form(default='')) -> dict[str, str | int]:
    '''Endpoint for receiving data to make prediction on. Data is converted into message and sent to consumer. Returns prediction result.'''
    
    uq_request_id = str(uuid.uuid4())
    message = _create_message(uq_request_id, data_input)
    await _send_request_to_consumer(message)
    prediction_result = await receive_result(uq_request_id)
    logger.debug(f'received prediction result: {prediction_result}')
    return prediction_result


async def main():
    await initialize_kafka()
    await start_server()
    await stop_kafka()


if __name__ == '__main__':
    logger.info('starting producer API...')
    asyncio.run(main())
