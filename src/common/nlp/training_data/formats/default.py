from src.common.nlp.training_data.formats import (
    JsonTrainingDataReader, TrainingDataWriter
)


class DefaultReader(JsonTrainingDataReader):
    def read_from_json(self, js, **kwargs):
        pass


class DefaultWriter(TrainingDataWriter):
    def dumps(self, training_data, **kwargs):
        pass
