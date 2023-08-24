import asyncio
import logging.config
from pathlib import Path

from aiokafka import ConsumerStoppedError
from aiokafka.structs import ConsumerRecord

from consumer_setup import initialize_kafka, stop_kafka, get_request_receiver, get_result_sender
from data_models import PredictionResultModel, FileChunkRequest, ResultResponse
from predictions.audio_preprocessors.audio_exceptions import CorruptedAudioFileException, AudioTooLongException, \
    AudioTooShortException
from predictions.models.base_prediction_model import BasePredictionModel
from predictions.models.prediction_features_model import PredictionFeaturesModel
from predictions.models.prediction_spectrograms_model import PredictionSpectrogramsModel
from tools.const_variables import FMA_OR_GTZAN, RESULTS_TOPIC, MIN_AUDIO_DURATION, MAX_AUDIO_DURATION
from tools.file_processing import create_file_from_chunks, fill_buffer, remove_files_after_error, remove_request_from_buffer

logging.config.fileConfig(Path('resources', 'logging.ini'), disable_existing_loggers=False)
logging.getLogger('aiokafka').setLevel(logging.ERROR)
logger = logging.getLogger('predictions')


def _create_prediction_result_message(request_id: str, prediction_result: PredictionResultModel) -> dict[str, str | float]:
    """Create prediction result message."""

    result_response = ResultResponse(request_id, 'success', prediction_result)
    return result_response.make_dict()


async def _handle_checksum_mismatch_error(request_id: str) -> None:
    """Send error result - checksum mismatch after concatenating chunks."""

    result_response = ResultResponse(request_id, 'fail', 'File checksum mismatch after sending for prediction.')
    result_sender = await get_result_sender()
    await result_sender.send_and_wait(RESULTS_TOPIC, result_response.make_dict())
    logger.warning(f'{request_id=} checksum mismatch after concatenating chunks')


async def _handle_file_corrupted_error(request_id: str) -> None:
    """Send error result - file cannot be loaded by librosa."""

    result_response = ResultResponse(request_id, 'fail', 'File corrupted, cannot make a prediction.')
    result_sender = await get_result_sender()
    await result_sender.send_and_wait(RESULTS_TOPIC, result_response.make_dict())
    logger.warning(f'{request_id=} file corrupted and cannot be loaded by librosa')
    remove_files_after_error()


async def _handle_audio_too_short_error(request_id: str) -> None:
    """Send error result - audio too short."""

    result_response = ResultResponse(request_id, 'fail', f'Audio too short, min audio duration = {MIN_AUDIO_DURATION} seconds')
    result_sender = await get_result_sender()
    await result_sender.send_and_wait(RESULTS_TOPIC, result_response.make_dict())
    remove_files_after_error()


async def _handle_audio_too_long_error(request_id: str) -> None:
    """Send error result - audio too long."""

    result_response = ResultResponse(request_id, 'fail', f'Audio too long, min audio duration = {MAX_AUDIO_DURATION} seconds')
    result_sender = await get_result_sender()
    await result_sender.send_and_wait(RESULTS_TOPIC, result_response.make_dict())
    remove_files_after_error()


async def _send_prediction_result(request_id: str, prediction_result: PredictionResultModel) -> None:
    """Send prediction result to API server."""
    
    result_response = _create_prediction_result_message(request_id, prediction_result)
    result_sender = await get_result_sender()
    await result_sender.send_and_wait(RESULTS_TOPIC, result_response)
    logger.info(f'result sent to producer: {result_response}')


async def _send_failed_prediction_response(request_id: str) -> None:
    """Send error result - prediction couldn't be made for some reason."""

    result_response = ResultResponse(request_id, 'fail', 'File could not be preprocessed to make prediction.')
    result_sender = await get_result_sender()
    await result_sender.send_and_wait(RESULTS_TOPIC, result_response)
    logger.warning(f'{request_id=} could not be preprocessed')


async def _perform_prediction_on_file(request_id: str, file_data: bytes, model: BasePredictionModel, file_extension: str) -> None:
    """Perform prediction on received file and send the results back to API server."""

    try:
        prediction_result = model.predict(request_id, file_data, file_extension)
    except AudioTooLongException:
        await _handle_audio_too_long_error(request_id)
    except AudioTooShortException:
        await _handle_audio_too_short_error(request_id)
    except CorruptedAudioFileException:
        await _handle_file_corrupted_error(request_id)
    else:
        if prediction_result:
            await _send_prediction_result(request_id, prediction_result)
            return
        await _send_failed_prediction_response(request_id)


async def process_messages(message: ConsumerRecord, model: BasePredictionModel) -> None:
    """Process messages received from producer, by putting them in requests buffer.
       If the whole file is received, make a prediction and send results back to API server."""

    message_content = message.value
    file_chunk_request = FileChunkRequest(**message_content)
    request_id = file_chunk_request.request_id
    fill_buffer(request_id, file_chunk_request)
    try:
        file_data = create_file_from_chunks(request_id, file_chunk_request.checksum)
    except BytesWarning:
        await _handle_checksum_mismatch_error(request_id)
        remove_request_from_buffer(request_id)
    else:
        if file_data:
            await _perform_prediction_on_file(request_id, file_data, model, file_chunk_request.file_extension)
            remove_request_from_buffer(request_id)


async def run_consumer() -> None:
    """Initialize Kafka and start consuming messages from API server."""

    await initialize_kafka()
    request_receiver = await get_request_receiver()
    model = PredictionSpectrogramsModel() if FMA_OR_GTZAN == 'FMA' else PredictionFeaturesModel()
    while True:
        try:
            message: ConsumerRecord = await request_receiver.getone()
            await process_messages(message, model)
        except ConsumerStoppedError:
            return  # TODO: error handling


async def main() -> None:
    await run_consumer()


if __name__ == '__main__':
    logger.info('starting server...')
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        if loop.is_closed():
            loop = asyncio.new_event_loop()
        loop.run_until_complete(stop_kafka())
