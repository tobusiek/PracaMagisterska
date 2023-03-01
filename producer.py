import json
import uuid

from fastapi import FastAPI
from kafka import KafkaProducer, KafkaConsumer
import uvicorn


# Initialize FastAPI app
app = FastAPI(debug=True)

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


# Define endpoint to receive data to be predicted
@app.post('/predict')
def predict(url: str) -> dict[str, str | int]:
    # Generate unique identifier for the request
    request_id = str(uuid.uuid4())

    # Create data for the request
    data = {'url': url}

    # Include unique identifier in message payload
    message = {"request_id": request_id, "data": data}

    # Send message to Kafka topic
    request_sender.send("requests_topic", message)

    # Wait for response on the results_topic
    for message in result_receiver:
        # Extract the response from the message payload
        response = message.value

        # Extract the request ID from the message payload
        response_id = response['request_id']

        # If the response ID matches the original request ID, return the predicted result
        if response_id == request_id:
            predicted_result = response['predicted_result']
            return predicted_result


if __name__ == '__main__':
    # Start the server
    uvicorn.run('producer:app', host='127.0.0.1', port=8081, reload=True)
