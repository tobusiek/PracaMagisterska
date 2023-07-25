from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass(frozen=True)
class FileChunkRequest:
    '''Dataclass for request with file chunk.'''

    request_id: str
    chunk_number: int
    num_of_chunks: int
    chunk_data: bytes
    file_extension: str
    checksum: str


@dataclass(frozen=True)
class PredictionResultModel:
    '''Dataclass for model's prediction result.'''

    first_genre: str
    first_genre_result: float
    second_genre: str
    second_genre_result: float
    # third_genre: str
    # third_genre_result: float

    def make_dict(self) -> dict[str, str | float]:
        return asdict(self)


@dataclass(frozen=True)
class ResultResponse:
    request_id: str
    status: str
    result: PredictionResultModel | str

    def make_dict(self) -> dict[str, str | float]:
        response = asdict(self)
        response.update(self.result.make_dict())
        return response


class PredictionModel(Protocol):
    def predict(self, request_id: str, file_data: bytes, file_extension: str) -> PredictionResultModel:
        ...
