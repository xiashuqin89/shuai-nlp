import os
import re
from typing import Dict, Text, Any, List, Optional

from shuai.common import (
    Message, TrainingData, TrainerModelConfig,
    logger, py_cloud_unpickle, py_cloud_pickle
)
from shuai.nlp.components import Component
from shuai.nlp.meta import Metadata
from shuai.nlp.constants import TEXT_FEATURES, FEATURIZER_REGEX
from ..featurizer import Featurizer


class CountVectorsFeaturizer(Featurizer):
    """
    Creates bag-of-words representation of intent features
    using sklearn's `CountVectorizer`.
    All tokens which consist only of digits (e.g. 123 and 99
    but not ab12d) will be represented by a single feature.
    Not adjust to chinese text, need to add tokenizer
    """
    name = FEATURIZER_REGEX
    provides = [TEXT_FEATURES]
    requires = []
    defaults = {
        "token_pattern": r'(?u)\b\w\w+\b',
        "strip_accents": None,
        "stop_words": None,
        "min_df": 1,
        "max_df": 1.0,
        "min_ngram": 1,
        "max_ngram": 1,
        "max_features": None
    }

    def __init__(self,
                 component_config: Dict[Text, Any] = None):
        super(CountVectorsFeaturizer, self).__init__(component_config)
        self.token_pattern = self.component_config['token_pattern']
        self.strip_accents = self.component_config['strip_accents']
        self.stop_words = self.component_config['stop_words']
        self.min_df = self.component_config['min_df']
        self.max_df = self.component_config['max_df']
        # set ngram range
        self.min_ngram = self.component_config['min_ngram']
        self.max_ngram = self.component_config['max_ngram']
        # limit vocabulary size
        self.max_features = self.component_config['max_features']
        # declare class instance for CountVector
        self.vector = None
        # preprocessor
        self.preprocessor = lambda s: re.sub(r'\b[0-9]+\b', 'NUMBER', s)

    @staticmethod
    def _lemmatize(message: Message):
        if message.get("spacy_doc"):
            return ' '.join([t.lemma_ for t in message.get("spacy_doc")])
        else:
            return message.text

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["sklearn"]

    def train(self,
              training_data: TrainingData,
              cfg: TrainerModelConfig = None,
              **kwargs):
        from sklearn.feature_extraction.text import CountVectorizer
        # use even single character word as a token
        # default config is enough
        self.vector = CountVectorizer(token_pattern=self.token_pattern,
                                      strip_accents=self.strip_accents,
                                      stop_words=self.stop_words,
                                      ngram_range=(self.min_ngram,
                                                   self.max_ngram),
                                      max_df=self.max_df,
                                      min_df=self.min_df,
                                      max_features=self.max_features,
                                      preprocessor=self.preprocessor)

        lem_exs = [self._lemmatize(example)
                   for example in training_data.intent_examples]
        # matrix for example input text
        # [0, 1, 1, 1, 0, 0, 1, 0, 1],
        # [0, 2, 0, 1, 0, 1, 1, 0, 1],
        # [1, 0, 0, 1, 1, 0, 1, 1, 1],
        # [0, 1, 1, 1, 0, 0, 1, 0, 1]]
        try:
            X = self.vector.fit_transform(lem_exs).toarray()
        except ValueError:
            self.vector = None
            return

        for i, example in enumerate(training_data.intent_examples):
            # create bag for each example
            example.set(TEXT_FEATURES, X[i])

    def process(self, message: Message, **kwargs):
        if self.vector is None:
            logger.error("There is no trained CountVectorizer: "
                         "component is either not trained or "
                         "didn't receive enough training data")
        else:
            bag = self.vector.transform([self._lemmatize(message)]).toarray()
            message.set(TEXT_FEATURES, bag)
