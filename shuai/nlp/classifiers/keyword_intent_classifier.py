import re
from typing import Dict, Text, Any, Optional

from shuai.common import Message
from shuai.nlp.components import Component
from shuai.nlp.constants import CLASSIFIER_KEYWORD, INTENT


class KeywordIntentClassifier(Component):
    name = CLASSIFIER_KEYWORD
    provides = [INTENT]

    def __init__(self,
                 component_config: Dict[Text, Any] = None,
                 intent_keyword_map: Optional[Dict] = None):
        super(KeywordIntentClassifier, self).__init__(component_config)
        self.component_config = component_config
        self.intent_keyword_map = intent_keyword_map or {}
        self.case_sensitive = self.component_config.get('case_sensitive')

    def process(self, message: Message, **kwargs):
        intent_name = self.parse(message.text)
        confidence = 0.0 if intent_name is None else 1.0
        intent = {"name": intent_name, "confidence": confidence}
        message.set(INTENT, intent, add_to_output=True)

    def parse(self, text: Text) -> Optional[Text]:
        re_flag = 0 if self.case_sensitive else re.IGNORECASE
        for keyword, intent in self.intent_keyword_map.items():
            if re.search(r'\b' + keyword + r'\b', text, flags=re_flag) or \
                    re.search(keyword, text, flags=re_flag):
                return intent
        return None
