from typing import Text, List

from shuai.nlp.components import Component
from shuai.nlp.constants import TOKENS, TOKENIZER_WHITESPACE
from .tokenizer import Tokenizer, Token


class WhitespaceTokenizer(Tokenizer, Component):
    name = TOKENIZER_WHITESPACE

    provides = [TOKENS]

    def tokenize(self, text: Text) -> List[Token]:
        """
        Split text by space
        Record the position
        """
        words = text.split()
        tokens, running_offset = [], 0
        for word in words:
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, word_offset))
        return tokens
