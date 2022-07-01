from typing import Text, Any

from src.common.nlp.training_data.formats import (
    TrainingDataReader, TrainingDataWriter
)


class MarkdownReader(TrainingDataReader):
    def read(self, filename: Text, **kwargs):
        pass

    def reads(self, s, **kwargs):
        pass


class MarkdownWriter(TrainingDataWriter):
    def dump(self, filename: Text, training_data: Any):
        pass

    def dumps(self, training_data):
        pass
