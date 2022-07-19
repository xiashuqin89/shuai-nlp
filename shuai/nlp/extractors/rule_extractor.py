import re
from typing import Dict, Text, Any, List, Optional

from shuai.common import (
    Message, TrainingData, TrainerModelConfig
)
from shuai.nlp.meta import Metadata
from shuai.nlp.constants import (
    EXTRACTOR_RULE, ENTITIES, TOKENS,
    ENTITY_ATTRIBUTE_TYPE, ENTITY_ATTRIBUTE_START, ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_VALUE
)
from .extractor import EntityExtractor


class RuleEntityExtractor(EntityExtractor):
    """
    support `xxxx ${param_a=axxxx}${param_b=bxxxx}`
    """
    name = EXTRACTOR_RULE
    provides = [ENTITIES]
    requires = [TOKENS]
    default = [
        'loc_match'
    ]

    def __init__(self,
                 component_config: Dict[Text, Any] = None,
                 rules: Optional[Dict] = None):
        super(RuleEntityExtractor, self).__init__(component_config)
        self.rules = rules or []
        self.rules.extend(self.default)

    @classmethod
    def _loc_match(cls, text: Text) -> List:
        entities = []
        matchers = re.finditer(r'\$\{.*?\=.*?\}', text)
        for matcher in matchers:
            start = matcher.start()
            end = matcher.end()
            name, value = text[start + 2: end - 1].split('=')
            entities.append({
                ENTITY_ATTRIBUTE_TYPE: name,
                ENTITY_ATTRIBUTE_START: start,
                ENTITY_ATTRIBUTE_END: end,
                ENTITY_ATTRIBUTE_VALUE: value
            })
        return entities

    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        entities = []
        for rule in self.rules:
            entities.extend(getattr(self, f'_{rule}')(message.text))

    def process(self, message: Message, **kwargs):
        extracted = self._extract_entities(message)
        message.set("entities",
                    message.get("entities", []) + extracted,
                    add_to_output=True)

    @classmethod
    def load(cls,
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional[EntityExtractor] = None,
             **kwargs) -> EntityExtractor:
        component_config = model_metadata.for_component(cls.name)
        return cls(component_config, model_metadata.get("rules"))
