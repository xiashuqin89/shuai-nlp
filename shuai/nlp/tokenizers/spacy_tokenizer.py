from typing import List

from spacy.tokens.doc import Doc

from shuai.common import TrainerModelConfig, TrainingData, Message
from shuai.nlp.constants import TOKENS
from .tokenizer import Tokenizer, Token


class SpacyTokenizer(Tokenizer):
    name = "tokenizer_spacy"
    provides = [TOKENS]

    def train(self,
              training_data: TrainingData,
              config: TrainerModelConfig,
              **kwargs):
        for example in training_data.training_examples:
            example.set(TOKENS, self.tokenize(example.get("spacy_doc")))

    def process(self, message: Message, **kwargs):
        message.set(TOKENS, self.tokenize(message.get("spacy_doc")))

    def tokenize(self, doc: Doc) -> List[Token]:
        return [Token(t.text, t.idx) for t in doc]
