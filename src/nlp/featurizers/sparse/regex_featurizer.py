import re
import os
import numpy as np
from typing import Dict, Text, Any, List, Optional

from src.common import (
    Message, TrainingData, TrainerModelConfig,
    read_json_file, write_json_to_file
)
from src.nlp.components import Component
from src.nlp.meta import Metadata
from src.nlp.constants import (
    TOKENS, TEXT_FEATURES, FEATURIZER_REGEX,
    REGEX_FEATURIZER_FILE_NAME
)
from ..featurizer import Featurizer


class RegexFeaturizer(Featurizer, Component):
    name = FEATURIZER_REGEX
    provides = [TEXT_FEATURES]
    requires = [TOKENS]

    def __init__(self,
                 component_config: Dict[Text, Any] = None,
                 known_patterns: List[Any] = None):
        super(RegexFeaturizer, self).__init__(component_config)
        self.known_patterns = known_patterns if known_patterns else []

    def features_for_patterns(self, message: Message) -> np.ndarray:
        """Tour input pattern list
        Try to match text find group
        If match, find tokens and add pattern attribute
        Return array [0.0, 1.0, 0.0, 0.0...] base on regex sort"""
        found = []
        for i, exp in enumerate(self.known_patterns):
            match = re.search(exp["pattern"], message.text)
            if match is not None:
                for t in message.get(TOKENS, []):
                    if t.offset < match.end() and t.end > match.start():
                        t.set("pattern", i)
                found.append(1.0)
            else:
                found.append(0.0)
        return np.array(found)

    def _text_features_with_regex(self, message: Message):
        if self.known_patterns is not None:
            extras = self.features_for_patterns(message)
            return self._combine_with_existing_text_features(message, extras)
        else:
            return message.get(TEXT_FEATURES)

    def train(self,
              training_data: TrainingData,
              config: TrainerModelConfig,
              **kwargs):
        """Add self define regex"""
        for example in training_data.regex_features:
            self.known_patterns.append(example)

        for example in training_data.training_examples:
            updated = self._text_features_with_regex(example)
            example.set("text_features", updated)

    def process(self, message: Message, **kwargs):
        updated = self._text_features_with_regex(message)
        message.set(TEXT_FEATURES, updated)

    @classmethod
    def load(cls,
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional[Featurizer] = None,
             **kwargs) -> Featurizer:
        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("regex_file", REGEX_FEATURIZER_FILE_NAME)
        regex_file = os.path.join(model_dir, file_name)

        if os.path.exists(regex_file):
            known_patterns = read_json_file(regex_file)
            return RegexFeaturizer(meta, known_patterns=known_patterns)
        else:
            return RegexFeaturizer(meta)

    def persist(self, model_dir: Text) -> Optional[Dict[Text, Any]]:
        if self.known_patterns:
            regex_file = os.path.join(model_dir, REGEX_FEATURIZER_FILE_NAME)
            write_json_to_file(regex_file, self.known_patterns, indent=4)
        return {"regex_file": REGEX_FEATURIZER_FILE_NAME}
