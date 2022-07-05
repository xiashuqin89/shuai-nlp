from typing import Text, Dict, Optional

from src.common.utils.stdlib import ordered


class Message(object):
    """
    Message is a unified data object that runs through
    the entire NLU workflow, so it also carries the
    intermediate results (such as word segmentation
    and part-of-speech tagging results) temporarily
    generated by each component of memory,
    as well as the final intent and entity information.
    The data stores the intent and entity information.
    When the subsequent components are processed,
    some variables will be added to the Message
    to store the intermediate results,
    that is, the responsibility of the set member method.
    """
    def __init__(self,
                 text: Text,
                 data: Dict[Optional] = None,
                 output_properties: Dict[Optional] = None,
                 time=None):
        self.text = text
        self.time = time
        self.data = data if data else {}

        if output_properties:
            self.output_properties = output_properties
        else:
            self.output_properties = set()

    def __eq__(self, other):
        if not isinstance(other, Message):
            return False
        else:
            return ((other.text, ordered(other.data)) ==
                    (self.text, ordered(self.data)))

    def __hash__(self):
        return hash((self.text, str(ordered(self.data))))

    def set(self, prop: Text, info: Optional, add_to_output=False):
        self.data[prop] = info
        if add_to_output:
            self.output_properties.add(prop)

    def get(self, prop: Text, default=None):
        return self.data.get(prop, default)

    def as_dict(self, only_output_properties: bool = False):
        if only_output_properties:
            d = {key: value
                 for key, value in self.data.items()
                 if key in self.output_properties}
        else:
            d = self.data
        return dict(d, text=self.text)

    @classmethod
    def build(cls, text: Text, intent=None, entities=None, id=None):
        data = {}
        if intent:
            data["intent"] = intent
        if entities:
            data["entities"] = entities
        if id and id.isdigit():
            data["id"] = int(id)
        return cls(text, data)
