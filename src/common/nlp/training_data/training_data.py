import os
from typing import Text, Dict, Any, Optional, List

from src.common.utils.io import read_file, write_to_file
from .formats.default import DefaultWriter
from .message import Message


class TrainingData(object):
    """Domain.yml"""

    def __init__(self,
                 training_examples: Optional[List[Message]] = None,
                 entity_synonyms: Optional[Dict[Text, Text]] = None,
                 regex_features: List = None):
        if training_examples:
            self.training_examples = self.sanitize_examples(training_examples)
        else:
            self.training_examples = []
        self.regex_features = regex_features if regex_features else []
        self.entity_synonyms = entity_synonyms if entity_synonyms else {}
        # self.sort_regex_features()
        # self.validate()
        # self.print_stats()

    def as_json(self, **kwargs) -> Text:
        return DefaultWriter().dumps(self)

    def persist(self, dir_name: Text) -> Dict[Text, Any]:
        """Persists this training data to disk"""
        data_file = os.path.join(dir_name, "training_data.json")
        write_to_file(data_file, self.as_json(indent=2))
        return {
            "training_data": "training_data.json"
        }

    @property
    def intent_examples(self) -> List[Message]:
        return [ex for ex in self.training_examples
                if ex.get("intent")]

    @property
    def entity_examples(self) -> List[Message]:
        return [ex for ex in self.training_examples
                if ex.get("entities")]

    @staticmethod
    def sanitize_examples(examples: List[Message]) -> List[Message]:
        """Makes sure the training data is clean.
        removes trailing whitespaces from intent annotations."""
        for ex in examples:
            if ex.get("intent"):
                ex.set("intent", ex.get("intent").strip())
        return examples
