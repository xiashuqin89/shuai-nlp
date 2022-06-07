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
        pass

    def as_json(self, **kwargs) -> Text:
        return DefaultWriter().dumps(self)

    def persist(self, dir_name: Text) -> Dict[Text, Any]:
        """Persists this training data to disk"""
        data_file = os.path.join(dir_name, "training_data.json")
        write_to_file(data_file, self.as_json(indent=2))
        return {
            "training_data": "training_data.json"
        }
