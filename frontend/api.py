import asyncio
import base64
from collections.abc import Generator
from dataclasses import dataclass
import logging.config
import math
import os
from pathlib import Path
from sys import getsizeof
from typing import BinaryIO
import uuid

from aiokafka.structs import ConsumerRecord
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi import Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.templating import _TemplateResponse
from uvicorn import Config, Server

from producer_setup import CHUNK_SIZE, MessageKey, initialize_kafka, stop_kafka, get_request_sender, get_result_receiver

LOGGER_PATH = Path('resources', 'logging.ini')
logging.config.fileConfig(LOGGER_PATH, disable_existing_loggers=False)
logging.getLogger('aiokafka').setLevel(logging.ERROR)
logger = logging.getLogger('fastapi')

app = FastAPI(debug=True)
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')


@dataclass
class FileChunk:
    chunk_number: int
    num_of_chunks: int
    chunk_data: bytes


async def start_server() -> None:
    '''Start FastAPI server.'''

    loop = asyncio.get_event_loop()
    config = Config(app, loop=loop, log_level='debug')
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
    logger.debug(f'request sent to consumer: {message}')


async def receive_result(request_id: str) -> dict[str, str | float]:
    '''Receive message with prediction result from consumer. Returns prediction result for specific request.'''

    result_receiver = await get_result_receiver()
    message: ConsumerRecord
    async for message in result_receiver:
        print('\n', type(message.value), message.value, '\n')
        response: dict[str, str] = message.value
        response_id = response['request_id']
        if response_id == request_id:
            try:
                return response
            except ValidationError as e:
                return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get('/', response_class=HTMLResponse)
async def root(request: Request) -> RedirectResponse:
    '''Endpoint for root, redirects to /prediction.'''

    return RedirectResponse(url='/predict', headers=request.headers)


@app.get('/favicon.ico', include_in_schema=False)
async def favicon() -> FileResponse:
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
    return templates.TemplateResponse('results.html', context)


def _encode_file_chunk_with_base64(file_chunk: bytes):
    return base64.b64encode(file_chunk).decode('utf-8')


def _create_message_with_file_chunk(request_id: str, file_chunk: FileChunk) -> dict[MessageKey, int | bytes]:
    message = {
            MessageKey.REQUEST_ID.value: request_id,
            MessageKey.CHUNK_NUMBER.value: file_chunk.chunk_number,
            MessageKey.NUM_CHUNKS.value: file_chunk.num_of_chunks,
            MessageKey.CHUNK_DATA.value: _encode_file_chunk_with_base64(file_chunk.chunk_data),
        }
    logger.debug(f'created new message with file chunk: {message}')
    return message


async def _chunkify_file(file: BinaryIO) -> Generator[FileChunk]:
    file_size = os.fstat(file.fileno()).st_size
    num_of_chunks = math.ceil(file_size / CHUNK_SIZE)
    file.seek(0)
    for chunk_number in range(num_of_chunks):
        chunk_data = file.read(CHUNK_SIZE)
        yield FileChunk(chunk_number, num_of_chunks, chunk_data)


@app.get('/predict_file', response_class=HTMLResponse)
async def get_predict_file(request: Request) -> _TemplateResponse:
    return templates.TemplateResponse('input-file-form.html', {'request': request})


@app.post('/predict_file')
async def post_predict_file(request: Request, file: UploadFile = File(...)):
    logger.debug(f'received file: {file.filename}')
    request_id = str(uuid.uuid4())
    file_object = file.file
    request_sender = await get_request_sender()
    async for file_chunk in _chunkify_file(file_object):
        message = _create_message_with_file_chunk(request_id, file_chunk)
        await request_sender.send_and_wait('requests_topic', message)
    return {'file': file.filename}


async def main() -> None:
    await initialize_kafka()
    await start_server()
    await stop_kafka()


if __name__ == '__main__':
    logger.info('starting producer API...')
    asyncio.run(main())
