from typing import Dict, Text, Any, List, Optional

from src.common import (
    TrainingData, TrainerModelConfig, Message,
    logger,
    PipelineRunningAbnormalError
)
from src.nlp.constants import (
    TEXT_FEATURES, FEATURIZER_TF_IDF, TOKENS, INTENT_FEATURES
)
from ..featurizer import Featurizer


"""
tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
train = [
    "Chinese Beijing Chinese",
    "Chinese Chinese Shanghai",
    "Chinese Macao",
    "Tokyo Japan Chinese"
]
tv_fit = tv.fit_transform(train)
print(tv.get_feature_names())
model = tv_fit.toarray()
x = model[3].reshape(1, -1)
print(x)

test = ["Chinese Chinese Chinese Tokyo Japan"]
test_fit = tv.transform(test)
y = test_fit.toarray()
print(y)

s = cosine_similarity(x, y)
print(s)
"""


class TfIdfVectorsFeaturizer(Featurizer):
    name = FEATURIZER_TF_IDF
    provides = [TEXT_FEATURES, INTENT_FEATURES]
    requires = [TOKENS]

    def __init__(self, component_config: Dict[Text, Any] = None):
        super(TfIdfVectorsFeaturizer, self).__init__(component_config)
        self.vector = None

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ['sklearn']

    @staticmethod
    def _transform_list2str(tokens: List[Text]):
        if not tokens:
            raise PipelineRunningAbnormalError('Need to do tokenizer before feature')
        return ' '.join([token.text[token.offset, token.end] for token in tokens])

    def train(self,
              training_data: TrainingData,
              cfg: TrainerModelConfig = None,
              **kwargs):
        """
        todo extract parameter and add stop_words
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vector = TfidfVectorizer(use_idf=True, smooth_idf=True)
        logger.info(f'tf-idf feature list {self.vector}')
        documents = [
            self._transform_list2str(example.get(TOKENS)) for example in training_data.intent_examples
        ]
        self.vector.fit_transform(documents)

    def process(self, message: Message, **kwargs):
        if not self.vector:
            logger.error('Tf-Idf matrix is not init')
        else:
            bag = self.vector.transform(self._transform_list2str(message.get(TOKENS))).toarray()
            message.set(TEXT_FEATURES, bag)
            message.set(INTENT_FEATURES, self.vector.toarray())
