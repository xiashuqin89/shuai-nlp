from collections import defaultdict
from typing import Dict, Text, Any

from shuai.common.utils.io import json_to_string
from .utils import transform_entity_synonyms
from .format import (
    JsonTrainingDataReader, TrainingDataWriter
)


class DefaultReader(JsonTrainingDataReader):
    def read_from_json(self, js, **kwargs):
        from shuai.common.nlp.training_data import TrainingData, Message
        validate_nlu_data(js)
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


def validate_nlu_data(data: Dict[Text, Any]):
    from jsonschema import validate
    from jsonschema import ValidationError

    try:
        validate(data, _nlu_data_schema())
    except ValidationError as e:
        e.message += (". Failed to validate training data, make sure your data "
                      "is valid. ")
        raise e


def _nlu_data_schema():
    training_example_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "intent": {"type": "string"},
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "value": {"type": "string"},
                        "entity": {"type": "string"}
                    },
                    "required": ["start", "end", "entity"]
                }
            }
        },
        "required": ["text"]
    }

    regex_feature_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "pattern": {"type": "string"},
        }
    }

    return {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {
                    "regex_features": {
                        "type": "array",
                        "items": regex_feature_schema
                    },
                    "common_examples": {
                        "type": "array",
                        "items": training_example_schema
                    },
                    "intent_examples": {
                        "type": "array",
                        "items": training_example_schema
                    },
                    "entity_examples": {
                        "type": "array",
                        "items": training_example_schema
                    }
                }
            }
        },
        "additionalProperties": False
    }
