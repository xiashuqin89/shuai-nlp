from collections import Counter
from importlib import import_module
from typing import (
    Dict, Text, Any, List, Tuple
)


import pandas as pd
import numpy as np

from shuai.common import (
    Message, TrainingData, TrainerModelConfig,
    logger
)
from shuai.nlp.constants import (
    CLASSIFIER_BM25, INTENT, TOKENS
)
from .classifier import Classifier


class BM25IntentClassifier(Classifier):
    """
    manual bm25 need to convert to sklearn
    algorithm: cos, levenshtein
    """
    name = CLASSIFIER_BM25
    provides = [INTENT]
    requires = [TOKENS]
    defaults = {
        'top_k': 1,
        'k1': 2,
        'k2': 1,
        'b': 0.75,
        'ignore_case': True,
        'algorithm': 'cos'
    }

    def __init__(self, component_config: Dict[Text, Any] = None, **kwargs):
        super(BM25IntentClassifier, self).__init__(component_config)
        self.top_k = self.component_config.get('top_k', 1)
        self.k1 = self.component_config.get('k1', 2)
        self.k2 = self.component_config.get('top_k', 1)
        self.b = self.component_config.get('b', 0.75)
        self.algorithm = self.component_config.get('algorithm', 'cos')
        self.ignore_case = bool(self.component_config.get('ignore_case', True))
        self.max_features = kwargs.get('max_features', None)
        self.top_keywords = None
        self.model = None

    def _get_top_keywords(self, word_matrix: List[Counter]) -> List[Text]:
        """
        :return top frequency words list
        todo need move to sparse featurizer
        """
        merge_tf = {}
        for _doc_tf in word_matrix:
            for k, v in _doc_tf.items():
                if k not in merge_tf:
                    merge_tf[k] = v
                else:
                    merge_tf[k] += v
        feature_name = sorted(merge_tf, key=merge_tf.__getitem__, reverse=True)
        logger.info("feature_name", feature_name)
        logger.info("merge_TF", merge_tf)
        if self.max_features:
            return feature_name[:self.max_features]
        return feature_name

    def _construct_count_vectors(self, word_matrix: List[Counter]) -> np.array:
        tf = [
            self._extract_count_vector(word_freq) for word_freq in word_matrix
        ]
        return np.array(tf)

    def _extract_count_vector(self, word_freq: Counter):
        return [word_freq.get(word, 0) for word in self.top_keywords]

    def _transform_score2confidence(self,
                                    intent: Dict[Text, Any],
                                    tf: np.array,
                                    query_tokens_freq_vector: np.array,
                                    query_text: Text):
        try:
            module = import_module(f'shuai.algorithm.{self.algorithm}')
        except ModuleNotFoundError:
            return None

        if self.algorithm == 'cos':
            return module.core_method(tf[intent['index']], list(query_tokens_freq_vector))
        elif self.algorithm == 'levenshtein':
            return module.core_method(intent['text'], query_text)

    def calculate(self, document_cnt: int, tf: np.array) -> Tuple:
        dl = np.sum(tf > 0, axis=1)
        avg_dl = sum(dl) / document_cnt
        df = np.sum(tf > 0, axis=0)
        idf = np.log((document_cnt - df + 0.5) / (df + 0.5) + 1)
        tmp = np.tile(self.k1 * (1 - self.b + self.b * dl / avg_dl).reshape((document_cnt, 1)),
                      (1, tf.shape[1]))
        score = np.tile(idf, (document_cnt, 1)) * (tf * (self.k1 + 1)) / (tf + tmp)
        return score, dl, avg_dl

    def train(self,
              training_data: TrainingData,
              cfg: TrainerModelConfig = None,
              **kwargs):

        clean_train_data = pd.DataFrame([
            {'text': eg.text, 'data': eg.data} for eg in training_data.intent_examples
        ])
        logger.info(f'read train conf, size={clean_train_data}')
        # clean_train_data = clean_train_data.drop_duplicates(keep='first').reset_index(drop=True)
        logger.info(f'removed duplication size={clean_train_data}')

        tokens_corpus: List[List[object]] = [eg.get(TOKENS) for eg in training_data.intent_examples]
        word_matrix = [Counter([token.text for token in tokens]) for tokens in tokens_corpus]
        self.top_keywords = self._get_top_keywords(word_matrix)

        tf = self._construct_count_vectors(word_matrix)
        document_cnt = len(clean_train_data)
        score, dl, avg_dl = self.calculate(document_cnt, tf)
        self.model = {
            'SCORE': score,
            'top_keywords': self.top_keywords,
            'data': clean_train_data,
            'tf': tf,
            'args': {'document_len': dl, 'avg_document_len': avg_dl, 'document_cnt': document_cnt}
        }

    def process(self, message: Message, **kwargs):
        segment = Counter([token.text for token in message.get(TOKENS)])
        intent = self.predict(segment, message.text)
        if intent:
            message.set(INTENT, {
                'name': intent['data'].get('intent'),
                'confidence': intent['confidence'],
                'text': intent['text']
            }, add_to_output=True)

    def predict(self, query_word_freq: Counter, query_text: Text) -> List:
        document_cnt = self.model.get('args').get('document_cnt')
        score = self.model.get('SCORE')
        tokens_freq_vector = np.array(self._extract_count_vector(query_word_freq))
        repeated_tokens_freq_vector = np.tile(tokens_freq_vector, (document_cnt, 1))
        tokens_vec_index = np.where(tokens_freq_vector > 0, 1, 0)
        corr_ratios = np.sum(
            np.tile(
                tokens_vec_index, (document_cnt, 1)
            ) * score * (repeated_tokens_freq_vector * (self.k2 + 1) / (repeated_tokens_freq_vector + self.k2)),
            axis=1
        )
        corr_records_index = np.nonzero(corr_ratios)[0]
        if len(corr_records_index) == 0:
            return []
        corr_records = self.model.get('data').loc[corr_records_index].reset_index()
        corr_records["score"] = corr_ratios[corr_records_index]
        records = corr_records.to_dict('records')
        records.sort(key=lambda x: x['score'], reverse=True)
        if records:
            intent = records[0]
            intent['confidence'] = self._transform_score2confidence(intent, self.model.get('tf'),
                                                                    tokens_freq_vector, query_text)
        else:
            intent = None
        return intent
