from typing import Optional

from .formats import TrainingDataReader, DomainFormatType


def _guess_format(filename: str) -> DomainFormatType:
    pass


def _load(filename: str, language: str = 'en') -> Optional[TrainingDataReader]:
    pass


def load_data(resource_name: str, language: str = 'en'):
    pass


def load_data_from_url(url: str, language: str = 'en'):
    pass
