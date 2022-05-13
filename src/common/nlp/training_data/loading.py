from typing import Optional

from .formats import TrainingDataReader, DomainFormatType


def _guess_format(filename: str) -> DomainFormatType:
    pass


def _load(filename: str, language: str = 'en') -> Optional[TrainingDataReader]:
    pass
