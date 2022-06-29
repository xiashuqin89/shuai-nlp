import os
from typing import (
    Dict, Text, Any, List, Tuple, Optional
)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np

from src.common import (
    Message, TrainingData, TrainerModelConfig,
    logger, py_cloud_pickle, py_cloud_unpickle
)
from src.nlp.meta import Metadata
from src.nlp.components import Component
from src.nlp.constants import (
    CLASSIFIER_SKLEARN, INTENT, RANKING, TEXT_FEATURES,
    INTENT_RANKING_LENGTH
)
from .classifier import Classifier


class SklearnIntentClassifier(Classifier):
    name = CLASSIFIER_SKLEARN
    provides = [INTENT, RANKING]
    requires = [TEXT_FEATURES]
    defaults = {
        "C": [1, 2, 5, 10, 20, 100],
        "kernels": ["linear"],
        "max_cross_validation_folds": 5
    }

    def __init__(self,
                 component_config: Dict[Text, Any] = None,
                 clf: GridSearchCV = None,
                 le: LabelEncoder = None):
        super(SklearnIntentClassifier, self).__init__(component_config)
        if le is not None:
            self.le = le
        else:
            self.le = LabelEncoder()
        self.clf = clf

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ['sklearn']

    def _num_cv_splits(self, y):
        folds = self.component_config["max_cross_validation_folds"]
        return max(2, min(folds, np.min(np.bincount(y)) // 5))

    def _create_classifier(self, num_threads: int, y):
        C = self.component_config['C']
        kernels = self.component_config['kernels']
        param_grid = [{'C': C, 'kernel': [str(k) for k in kernels]}]
        cv_splits = self._num_cv_splits(y)

        return GridSearchCV(SVC(C=1,
                                probability=True,
                                class_weight='balanced'),
                            param_grid=param_grid,
                            n_jobs=num_threads,
                            cv=cv_splits,
                            scoring='f1_weighted',
                            verbose=1)

    def transform_labels_str2num(self, labels: List[Text]) -> np.ndarray:
        """Transforms a list of strings into numeric label representation.
        return: array([0, 1, 2])"""
        return self.le.fit_transform(labels)

    def transform_labels_num2str(self, y) -> np.ndarray:
        """Transforms a list of strings into numeric label representation.
        return array(['intent1', 'intent2'])"""
        return self.le.inverse_transform(y)

    def train(self,
              training_data: TrainingData,
              cfg: TrainerModelConfig,
              **kwargs):
        num_threads = kwargs.get('num_threads', 1)
        labels = [e.get('intent') for e in training_data.intent_examples]
        if len(set(labels)) < 2:
            logger.warn('Can not train an intent classifier. '
                        'Need at least 2 different classes. '
                        'Skipping training of intent classifier.')
            return

        y = self.transform_labels_str2num(labels)
        X = np.stack([example.get(TEXT_FEATURES)
                      for example in training_data.intent_examples])
        self.clf = self._create_classifier(num_threads, y)
        self.clf.fit(X, y)

    def process(self, message: Message, **kwargs):
        if not self.clf:
            intent = None
            intent_ranking = []
        else:
            X = message.get(TEXT_FEATURES).reshape(1, -1)
            intent_ids, probabilities = self.predict(X)
            intents = self.transform_labels_num2str(intent_ids)
            intents, probabilities = intents.flatten(), probabilities.flatten()
            if intents.size > 0 and probabilities.size > 0:
                ranking = list(zip(list(intents),
                                   list(probabilities)))[:INTENT_RANKING_LENGTH]
                intent = {'name': intents[0], 'confidence': probabilities[0]}
                intent_ranking = [{'name': intent_name, 'confidence': score}
                                  for intent_name, score in ranking]
            else:
                intent = {"name": None, "confidence": 0.0}
                intent_ranking = []

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def predict(self, X: List) -> Tuple[np.ndarray, np.ndarray]:
        """Call on the estimator with the best found parameters"""
        pred_result = self.clf.predict_proba(X)
        sorted_indices = np.fliplr(np.argsort(pred_result, axis=1))
        return sorted_indices, pred_result[:, sorted_indices]

    def persist(self, model_dir: Text) -> Optional[Dict[Text, Any]]:
        file_name = f'{CLASSIFIER_SKLEARN}.pkl'
        classifier_file = os.path.join(model_dir, file_name)
        py_cloud_pickle(classifier_file, self)
        return {"classifier_file": file_name}

    @classmethod
    def load(cls,
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional[Component] = None,
             **kwargs):
        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("classifier_file", f'{CLASSIFIER_SKLEARN}.pkl')
        classifier_file = os.path.join(model_dir, file_name)

        if os.path.exists(classifier_file):
            return py_cloud_unpickle(classifier_file)
        else:
            return cls(meta)
