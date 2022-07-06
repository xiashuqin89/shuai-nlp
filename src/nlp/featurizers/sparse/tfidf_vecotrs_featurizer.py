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
    defaults = {
        "use_idf": True,
        "smooth_idf": True,
        "stop_words": None,
        "max_df": 1.0,
        "min_df": 1
    }

    def __init__(self, component_config: Dict[Text, Any] = None):
        super(TfIdfVectorsFeaturizer, self).__init__(component_config)
        self.use_idf = self.component_config.get('use_idf', True)
        self.smooth_idf = self.component_config.get('smooth_idf', True)
        self.stop_words = self.component_config.get('stop_words')
        self.max_df = self.component_config.get('max_df', 1.0)
        self.min_df = self.component_config.get('min_df', 1)
        self.vector = None

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ['sklearn']

    @staticmethod
    def _transform_list2str(tokens: List[Text]):
        if not tokens:
            raise PipelineRunningAbnormalError('Need to do tokenizer before feature')
        return ' '.join([token.text[token.offset: token.end] for token in tokens])

    def train(self,
              training_data: TrainingData,
              cfg: TrainerModelConfig = None,
              **kwargs):
        """
        todo extract parameter and add stop_words
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vector = TfidfVectorizer(use_idf=self.use_idf, smooth_idf=self.smooth_idf,
                                      stop_words=self.stop_words,
                                      max_df=self.max_df, min_df=self.min_df)
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
