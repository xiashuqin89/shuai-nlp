from typing import Text, Optional

from src.common import Message, TrainingData, TrainerModelConfig
from src.nlp.meta import Metadata
from src.nlp.components import Component


class Classifier(Component):
    def train(self,
              training_data: TrainingData,
              cfg: TrainerModelConfig,
              **kwargs):
        pass

    def process(self, message: Message, **kwargs):
        pass

    def persist(self, model_dir: Text):
        pass

    @classmethod
    def load(cls,
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional[Component] = None,
             **kwargs):
        pass
