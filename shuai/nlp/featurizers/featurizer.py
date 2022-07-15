import os
from typing import Text, Dict, Any, Optional

import numpy as np

from shuai.common import (
    Message,
    py_cloud_pickle, py_cloud_unpickle, logger
)
from shuai.nlp.constants import TEXT_FEATURES
from shuai.nlp.components import Component
from shuai.nlp.meta import Metadata


class Featurizer(Component):
    @staticmethod
    def _combine_with_existing_text_features(message: Message,
                                             additional_features: np.ndarray):
        if message.get(TEXT_FEATURES) is not None:
            return np.hstack((message[TEXT_FEATURES], additional_features))
        else:
            return additional_features

    @classmethod
    def load(cls,
             model_dir: Text = None,
             model_metadata: Metadata = None,
             cached_component: Optional[Component] = None,
             **kwargs):
        meta = model_metadata.for_component(cls.name)
        if model_dir and meta.get('featurizer_file'):
            file_name = meta['featurizer_file']
            featurizer_file = os.path.join(model_dir, file_name)
            return py_cloud_unpickle(featurizer_file)
        else:
            logger.warning("Failed to load featurizer. Maybe path {} "
                           "doesn't exist".format(os.path.abspath(model_dir)))
            return cls(meta)

    def persist(self, model_dir: Text) -> Dict[Text, Any]:
        featurizer_file = os.path.join(model_dir, self.name + ".pkl")
        py_cloud_pickle(featurizer_file, self)
        return {"featurizer_file": self.name + ".pkl"}
