from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FileChunkRequest:
    """Dataclass for request with file chunk."""

    request_id: str
    chunk_number: int
    num_of_chunks: int
    chunk_data: str
    file_extension: str
    checksum: str


@dataclass(frozen=True)
class PredictionResultModel:
    """Dataclass for model's prediction result."""

    first_genre: str
    first_genre_result: float
    second_genre: str
    second_genre_result: float


@dataclass(frozen=True)
class ResultResponse:
    """Dataclass for response with result."""

    request_id: str
    status: str
    result: PredictionResultModel | str

    def make_dict(self) -> dict[str, str | float]:
        return asdict(self)
