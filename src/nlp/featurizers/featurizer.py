import numpy as np

from src.common import Message
from src.nlp.constants import TEXT_FEATURES


class Featurizer:
    @staticmethod
    def _combine_with_existing_text_features(message: Message,
                                             additional_features: np.ndarray):
        if message.get(TEXT_FEATURES) is not None:
            return np.hstack((message[TEXT_FEATURES], additional_features))
        else:
            return additional_features
