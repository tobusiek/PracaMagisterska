import asyncio
import logging.config
from pathlib import Path
import uuid

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.templating import _TemplateResponse
from uvicorn import Config, Server

from message_processing import get_file_data, receive_prediction_result, send_file_chunks
from producer_setup import initialize_kafka, stop_kafka

LOGGER_PATH = Path('resources', 'logging.ini')
logging.config.fileConfig(LOGGER_PATH, disable_existing_loggers=False)
logging.getLogger('aiokafka').setLevel(logging.ERROR)
logger = logging.getLogger('fastapi')

app = FastAPI(debug=True)
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')


async def start_server() -> None:
    """Start FastAPI server."""

    loop = asyncio.get_event_loop()
    config = Config(app, loop=loop, log_level='debug')
    server = Server(config)
    logger.info('starting server...')
    await server.serve()
    logger.info('server stopped successfully')


@app.get('/', response_class=HTMLResponse)
async def root(request: Request) -> RedirectResponse:
    """Endpoint for root, redirects to /predict."""

    return RedirectResponse(url='/predict', headers=request.headers)


@app.get('/favicon.ico', include_in_schema=False)
async def favicon() -> FileResponse:
    """Endpoint for favicon.ico."""

    return FileResponse(Path('static', 'images', 'favicon.svg'))


@app.get('/predict', response_class=HTMLResponse)
async def get_predict(request: Request) -> _TemplateResponse:
    """Endpoint for getting the form to upload file for prediction with GET."""

    return templates.TemplateResponse('input-file-form.html', {'request': request})


@app.post('/predict')
async def post_predict(request: Request, file: UploadFile = File(...)) -> _TemplateResponse:
    """Endpoint for uploading an audio file for prediction with POST."""
    
    request_id = str(uuid.uuid4())
    file_data = await get_file_data(file, request_id)
    logger.debug(f'received file: {file_data.filename} for {request_id=}')
    await send_file_chunks(file_data, request_id)
    results = await receive_prediction_result(request_id)
    logger.info(f'{request_id=} received results: {results}')
    context = {'request': request, **results}
    return templates.TemplateResponse('results.html', context)


async def main() -> None:
    await initialize_kafka()
    await start_server()
    await stop_kafka()


if __name__ == '__main__':
    logger.info('starting producer API...')
    asyncio.run(main())
