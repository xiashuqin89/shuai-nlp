from typing import (
    Dict, Text, Any, List, Tuple
)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from shuai.common import (
    Message, TrainingData, TrainerModelConfig,
    logger
)
from shuai.nlp.constants import (
    INTENT, RANKING, TEXT_FEATURES, CLASSIFIER_TF_IDF,
    INTENT_RANKING_LENGTH, INTENT_FEATURES
)
from .classifier import Classifier


class TfIdfIntentClassifier(Classifier):
    name = CLASSIFIER_TF_IDF
    provides = [INTENT, RANKING]
    requires = [TEXT_FEATURES, INTENT_FEATURES]

    def __init__(self, component_config: Dict[Text, Any] = None):
        super(TfIdfIntentClassifier, self).__init__(component_config)
        self.intents = None
        self.matrix = None

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ['sklearn']

    def train(self,
              training_data: TrainingData,
              cfg: TrainerModelConfig = None,
              **kwargs):
        self.intents = training_data.intent_examples

    def process(self, message: Message, **kwargs):
        """
        Algorithmic complexity is not good
        need to promote
        """
        self.matrix = message.get(INTENT_FEATURES)
        if self.matrix is None:
            intent = None
            intent_ranking = []
        else:
            X = message.get(TEXT_FEATURES).reshape(1, -1)
            intent_ids, probabilities = self.predict(X)
            if intent_ids is None:
                intent = {"name": None, "confidence": 0.0}
                intent_ranking = []
            else:
                max_index = np.argmax(probabilities)
                intent = {'name': self.intents[max_index].get('intent'),
                          'confidence': probabilities[max_index]}
                ranking = list(zip(list(intent_ids), list(probabilities)))
                intent_ranking = [{'name': self.intents[index].get('intent'), 'confidence': score}
                                  for index, score in ranking]
                intent_ranking.sort(key=lambda x: x['confidence'], reverse=True)
                intent_ranking = intent_ranking[:INTENT_RANKING_LENGTH]

        message.set(INTENT, intent, add_to_output=True)
        message.set(RANKING, intent_ranking, add_to_output=True)

    def predict(self, X: List) -> Tuple[np.ndarray, np.ndarray]:
        try:
            cos_values = [cosine_similarity(item.reshape(1, -1), X)[0][0] for item in self.matrix]
            cos_values = np.array(cos_values)
        except (ValueError, IndexError):
            logger.error('Input text feature is not correct')
            return None

        sorted_indices = np.argsort(cos_values)
        return sorted_indices, cos_values
