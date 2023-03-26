import asyncio
import json
import logging
import logging.config
from pathlib import Path
import uuid

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi import Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from uvicorn import Config, Server

# Set up logging
logging.config.fileConfig(Path('resources', 'logging.ini'), disable_existing_loggers=False)
logging.getLogger('aiokafka').setLevel(logging.ERROR)
logger = logging.getLogger('debug')

# Set up app after setting up logging
app = FastAPI(debug=True)
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

request_sender = None
result_receiver = None


async def create_producer() -> AIOKafkaProducer:
    return AIOKafkaProducer(
        bootstrap_servers='localhost:9092', 
        value_serializer=lambda x: json.dumps(x).encode('utf-8'),
    )


async def get_producer() -> AIOKafkaProducer:
    return request_sender if request_sender else await create_producer()


async def create_consumer() -> AIOKafkaConsumer:
    return AIOKafkaConsumer(
        'results_topic',
        bootstrap_servers='localhost:9092',
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id='results-group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_commit_interval_ms=1000
    )


async def get_consumer() -> AIOKafkaConsumer:
    return result_receiver if result_receiver else await create_consumer()


async def stop_kafka(request_sender: AIOKafkaProducer, result_receiver: AIOKafkaConsumer) -> None:
    await request_sender.stop()
    await result_receiver.stop()


# Define the endpoint to root
@app.get('/', response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse(url='/predict', headers=request.headers)


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(os.path.join('static', 'images', 'favicon.svg'))


# Define the endpoint to submit a request
@app.get('/predict', response_class=HTMLResponse)
async def get_predict(request: Request):
    return templates.TemplateResponse('input-form.html', {'request': request})


# Define endpoint to receive data to be predicted
@app.post('/predict')
async def post_predict(data_input: int | str = Form(default='')) -> dict[str, str | int]:
    # Generate unique identifier for the request
    request_id = str(uuid.uuid4())

    # Include unique identifier in message payload
    message = {'request_id': request_id, 'data': data_input}

    # Send message to Kafka topic
    request_sender = await get_producer()
    await request_sender.send_and_wait('requests_topic', message)
    await request_sender.flush()
    logger.debug(f'request sent to consumer: {message}')

    # Wait for response on the results_topic
    result = await receive_result(request_id)

    return result


# Define function to receive result from Kafka topic
async def receive_result(request_id: str) -> dict[str, str | int]:
    async for message in result_receiver:
        # Extract the response from the message payload
        response = message.value

        # Extract the request ID from the message payload
        response_id = response['request_id']

        # If the response ID matches the original request ID, return the predicted result
        if response_id == request_id:
            try:
                predicted_result = response['predicted_result']
                result = {'predicted_result': predicted_result}
                return result
            except ValidationError as e:
                return JSONResponse(content={"error": str(e)}, status_code=400)


async def main():
    global request_sender, result_receiver
    request_sender = await create_producer()
    logger.debug(f'request_sender created {request_sender}')
    result_receiver = await create_consumer()
    logger.debug(f'result_receiver created {result_receiver}')
    loop = asyncio.get_event_loop()
    config = Config(app, loop=loop)
    server = Server(config)
    logger.debug('starting server')
    await server.serve()
    while True:
        try:
            ...
        except KeyboardInterrupt:
            logger.info('API interrupted by user')
            loop.run_until_complete(stop_kafka)
            loop.stop()
            return


if __name__ == '__main__':
    logger.info('API started')
    asyncio.run(main())
