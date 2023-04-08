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
from starlette.templating import _TemplateResponse
from uvicorn import Config, Server

from producer_setup import initialize_kafka, stop_kafka, get_request_sender, get_result_receiver

LOGGER_PATH = Path('resources', 'logging.ini')
logging.config.fileConfig(LOGGER_PATH, disable_existing_loggers=False)
logging.getLogger('aiokafka').setLevel(logging.ERROR)
logger = logging.getLogger('fastapi')

app = FastAPI(debug=True)
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

request_sender: AIOKafkaProducer = None
result_receiver: AIOKafkaConsumer = None


async def start_server() -> None:
    '''Start FastAPI server.'''

    loop = asyncio.get_event_loop()
    config = Config(app, loop=loop, log_level='debug', reload=True)
    server = Server(config)
    logger.debug('starting server...')
    await server.serve()
    logger.debug('server stopped')


def _create_message(request_id: str, data: str | int) -> dict[str, str | float]:
    '''Create message to be sent to the consumer.'''

    message = {'request_id': request_id, 'data': data}
    logger.debug(f'created new message: {message}')
    return message


async def _send_request_to_consumer(message: dict[str, str | float]) -> None:
    '''Send prediction request to consumer.'''

    request_sender = await get_request_sender()
    await request_sender.send_and_wait('requests_topic', message)
    await request_sender.flush()  # TODO try removing
    logger.debug(f'request sent to consumer: {message}')


async def receive_result(request_id: str) -> dict[str, str | float]:
    '''Receive message with prediction result from consumer. Returns prediction result for specific request.'''

    result_receiver = await get_result_receiver()
    message: ConsumerRecord
    async for message in result_receiver:
        response: dict[str, str] = json.loads(message.value)
        response_id = response['request_id']
        if response_id == request_id:
            try:
                response.pop('request_id', None)
                return response
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
async def get_predict(request: Request) -> _TemplateResponse:
    '''Endpoint for making prediction requests.'''
    return templates.TemplateResponse('input-form.html', {'request': request})


@app.post('/predict', response_class=HTMLResponse)
async def post_predict(request: Request, data_input: str | int = Form(default='')) -> _TemplateResponse:
    '''Endpoint for receiving data to make prediction on.
    Data is converted into message and sent to consumer.
    Returns prediction result.'''
    
    uq_request_id = str(uuid.uuid4())
    message = _create_message(uq_request_id, data_input)
    await _send_request_to_consumer(message)
    prediction_result = await receive_result(uq_request_id)
    logger.debug(f'received prediction result: {prediction_result}')
    context = {'request': request, **prediction_result}
    print('\n', context, '\n')
    return templates.TemplateResponse('results.html', context)


async def main():
    await initialize_kafka()
    await start_server()
    await stop_kafka()


if __name__ == '__main__':
    logger.info('starting producer API...')
    asyncio.run(main())
