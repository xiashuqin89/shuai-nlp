import json
from typing import Optional

import requests

from src.common.nlp.training_data.training_data import TrainingData
from src.common.utils.io import read_file, list_files
from src.common.utils.stdlib import is_url
from src.common.constants import MARKDOWN_AVAILABLE_SECTIONS
from src.common.log import logger
from .formats import TrainingDataReader, DomainFormatType
from .formats.default import DefaultReader
from .formats.markdown import MarkdownReader


def _guess_format(filename: str):
    """
    Default support json, if parse failed,
    Turned to markdown
    """
    def _markdown_section_markers():
        return ["## {}:".format(s) for s in MARKDOWN_AVAILABLE_SECTIONS]

    guess = DomainFormatType.DEFAULT
    content = read_file(filename)
    try:
        json.loads(content)
    except ValueError:
        if any([marker in content for marker in _markdown_section_markers]):
            guess = DomainFormatType.MARKDOWN
        else:
            guess = DomainFormatType.UNK
    return guess


def _reader_factory(f_format: DomainFormatType):
    reader = None
    if f_format == DomainFormatType.DEFAULT:
        reader = DefaultReader()
    elif f_format == DomainFormatType.MARKDOWN:
        reader = MarkdownReader()
    return reader


def _load(filename: str, language: str = 'en') -> Optional[TrainingDataReader]:
    f_format = _guess_format(filename)
    if f_format == DomainFormatType.UNK:
        raise ValueError("Unknown data format for file {}".format(filename))

    logger.info("Training data format of {} is {}".format(filename, f_format))
    reader = _reader_factory(f_format)

    if reader:
        return reader.read(filename, language=language, fformat=f_format)
    else:
        return None


def load_data(resource_name: str, language: str = 'en') -> TrainingData:
    files = list_files(resource_name)
    data_sets = [_load(f, language) for f in files]
    data_sets = [ds for ds in data_sets if ds]
    if len(data_sets) == 0:
        return TrainingData()
    elif len(data_sets) == 1:
        return data_sets[0]
    else:
        return data_sets[0].merge(*data_sets[1:])


def load_data_from_url(url: str, language: str = 'en') -> TrainingData:
    """
    Load data from net using api
    Only support json format
    """
    # todo need to validate url and response body
    if not is_url(url):
        raise requests.exceptions.InvalidURL(url)

    response = requests.get(url)
    return DefaultReader().read_from_json(response)
