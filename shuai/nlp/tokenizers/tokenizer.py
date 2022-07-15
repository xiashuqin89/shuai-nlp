import abc
from typing import Text

from shuai.common import Message, TrainingData, TrainerModelConfig
from shuai.nlp.constants import TOKENS
from shuai.nlp.components import Component


class Token(object):
    def __init__(self, text, offset, data=None):
        self.offset = offset
        self.text = text
        self.end = offset + len(text)
        self.data = data if data else {}

    def __eq__(self, other):
        if not isinstance(other, Token):
            return False
        return (self.offset, self.end, self.text) == (other.offset, other.end, other.text)

    def set(self, prop, info):
        self.data[prop] = info

    def get(self, prop, default=None):
        return self.data.get(prop, default)

    def fingerprint(self):
        pass


class Tokenizer(Component):
    def process(self, message: Message, **kwargs):
        message.set(TOKENS, self.tokenize(message.text))

    def train(self,
              training_data: TrainingData,
              config: TrainerModelConfig,
              **kwargs):
        for example in training_data.training_examples:
            example.set(TOKENS, self.tokenize(example.text))

    @abc.abstractmethod
    def tokenize(self, text: Text):
        pass
