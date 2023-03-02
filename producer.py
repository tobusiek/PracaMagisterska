import json
import logging
import logging.config
import os
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi import Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from kafka import KafkaProducer, KafkaConsumer
from pydantic import ValidationError

# Set up logging
logging.config.fileConfig(os.path.join('resources', 'logging.ini'), disable_existing_loggers=False)
logging.getLogger('kafka').setLevel(logging.ERROR)
logger = logging.getLogger('debug')

# Set up app after setting up logging
app = FastAPI(debug=True)
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

# Initialize Kafka producer
request_sender = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# Initialize Kafka consumer
result_receiver = KafkaConsumer(
    'results_topic',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='requests-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)


# Define the endpoint to root
@app.get('/')
async def root(request: Request):
    data_input = await request.form()
    logger.info(f'{data_input=}')
    return {"data_output": data_input}


# Define the endpoint to submit a request
@app.get('/predict', response_class=HTMLResponse)
async def get_predict(request: Request):
    return templates.TemplateResponse('input-form.html', {'request': request})


# Define endpoint to receive data to be predicted
@app.post('/predict')
async def post_predict(data_input: int | str = Form(...)) -> dict[str, str | int]:
    # Generate unique identifier for the request
    request_id = str(uuid.uuid4())

    # Include unique identifier in message payload
    message = {'request_id': request_id, 'data': data_input}

    # Send message to Kafka topic
    request_sender.send('requests_topic', message)
    logger.debug(f'request sent to consumer: {message}')

    # Wait for response on the results_topic
    result = await receive_result(request_id)

    return result


# Define function to receive result from Kafka topic
async def receive_result(request_id: str) -> dict[str, str | int]:
    for message in result_receiver:
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
