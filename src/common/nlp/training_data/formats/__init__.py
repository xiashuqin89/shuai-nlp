from enum import Enum


class DomainFormatType(Enum):
    WIT = "wit"
    LUIS = "luis"
    UNK = "unk"
    MARKDOWN = "md"


class TrainingDataReader(object):
    pass
