import os
from typing import (
    Dict, Text, Any, List, Tuple, Optional
)

from src.common import (
    TrainingData, Message, TrainerModelConfig,
    as_text_type, write_json_to_file, read_json_file
)
from src.nlp.constants import (
    ENTITIES, EXTRACTOR_SYNONYMS, ENTITY_SYNONYMS_FILE_NAME
)
from src.nlp.meta import Metadata
from src.nlp.components import Component
from .extractor import EntityExtractor


class EntitySynonymMapper(EntityExtractor):
    name = EXTRACTOR_SYNONYMS
    provides = [ENTITIES]

    def __init__(self,
                 component_config: Optional[Dict[Text, Text]] = None,
                 synonyms: Dict = None):
        super(EntitySynonymMapper, self).__init__(component_config)
        self.synonyms = synonyms if synonyms else {}

    def _add_entities_if_synonyms(self, entity_a: Text, entity_b: Text):
        if entity_b is None:
            return
        original = as_text_type(entity_a)
        replacement = as_text_type(entity_b)
        if original == replacement:
            return
        original = original.lower()
        self.synonyms[original] = replacement

    def _replace_synonyms(self, entities: List[Dict]):
        for entity in entities:
            entity_value = str(entity["value"])
            if entity_value.lower() in self.synonyms:
                entity["value"] = self.synonyms[entity_value.lower()]
                self._add_processor_name(entity)

    def train(self,
              training_data: TrainingData,
              cfg: TrainerModelConfig = None,
              **kwargs):
        for key, value in list(training_data.entity_synonyms.items()):
            self._add_entities_if_synonyms(key, value)

        for example in training_data.entity_examples:
            for entity in example.get(ENTITIES, []):
                entity_val = example.text[entity["start"]:entity["end"]]
                self._add_entities_if_synonyms(entity_val, str(entity.get("value")))

    def process(self, message: Message, **kwargs):
        updated_entities = message.get(ENTITIES, [])[:]
        self._replace_synonyms(updated_entities)
        message.set("entities", updated_entities, add_to_output=True)

    def persist(self, model_dir: Text) -> Optional[Dict[Text, Any]]:
        if self.synonyms:
            entity_synonyms_file = os.path.join(model_dir,
                                                ENTITY_SYNONYMS_FILE_NAME)
            write_json_to_file(entity_synonyms_file, self.synonyms,
                               separators=(',', ': '))
        return {"synonyms_file": ENTITY_SYNONYMS_FILE_NAME}

    @classmethod
    def load(cls,
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional[Component] = None,
             **kwargs):
        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("synonyms_file", ENTITY_SYNONYMS_FILE_NAME)
        entity_synonyms_file = os.path.join(model_dir, file_name)

        synonyms = None
        if os.path.isfile(entity_synonyms_file):
            synonyms = read_json_file(entity_synonyms_file)
        return cls(meta, synonyms)
