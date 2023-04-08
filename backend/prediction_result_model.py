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
        return self.__dict__
