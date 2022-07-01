from collections import defaultdict

from src.common.nlp.training_data.formats import (
    JsonTrainingDataReader, TrainingDataWriter
)
from src.common.nlp.training_data import transform_entity_synonyms
from src.common.nlp.training_data.training_data import TrainingData
from src.common.nlp.training_data.message import Message
from src.common.utils.io import json_to_string


class DefaultReader(JsonTrainingDataReader):
    def read_from_json(self, js, **kwargs):
        # todo need to validate file format
        data = js['data']
        common_examples = data.get("common_examples", [])
        intent_examples = data.get("intent_examples", [])
        entity_examples = data.get("entity_examples", [])
        entity_synonyms = data.get("entity_synonyms", [])
        regex_features = data.get("regex_features", [])

        entity_synonyms = transform_entity_synonyms(entity_synonyms)
        all_examples = common_examples + intent_examples + entity_examples

        training_examples = [
            Message.build(ex['text'], ex.get("intent"), ex.get("entities"), ex.get("id"))
            for ex in all_examples
        ]
        return TrainingData(training_examples, entity_synonyms, regex_features)


class DefaultWriter(TrainingDataWriter):
    def dumps(self, training_data, **kwargs):
        js_entity_synonyms = defaultdict(list)
        for k, v in training_data.entity_synonyms.items():
            if k != v:
                js_entity_synonyms[v].append(k)

        formatted_synonyms = [{'value': value, 'synonyms': synonyms}
                              for value, synonyms in js_entity_synonyms.items()]
        formatted_examples = [example.as_dict()
                              for example in training_data.training_examples]

        return json_to_string({
            'data': {
                'common_examples': formatted_examples,
                'regex_features': training_data.regex_features,
                'entity_synonyms': formatted_synonyms
            }
        }, **kwargs)
