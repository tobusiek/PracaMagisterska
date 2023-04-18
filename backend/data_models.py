import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionResultModel:
    request_id: str
    first_genre: str
    first_genre_result: float
    second_genre: str
    second_genre_result: float
    third_genre: str
    third_genre_result: float

    def make_dict(self) -> dict[str, str | float]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class FileChunkRequest:
    request_id: str
    chunk_number: int
    num_of_chunks: int
    chunk_data: bytes
    file_extension: str
