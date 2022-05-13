from typing import Text, Optional, List

from src.common import TrainerModelConfig, TrainingData
from src.nlp import Persistor, ComponentBuilder, Component


class Trainer(object):
    """
    Trainer will load the data and train all components.
    Requires a pipeline specification(config.yml) and configuration(domain.yml)
    to use for the training.
    """

    SUPPORTED_LANGUAGES = ["zh", "en"]

    def __init__(self,
                 cfg: TrainerModelConfig,
                 component_builder: Optional[ComponentBuilder] = None,
                 skip_validation: bool = False):

        self.config = cfg
        self.skip_validation = skip_validation
        self.training_data = None

    def train(self, data: TrainingData, **kwargs):
        pass

    @staticmethod
    def _build_pipeline(cfg: TrainerModelConfig,
                        component_builder: ComponentBuilder) -> List:
        """Transform the passed names of the pipeline components into classes"""
        pipeline = [
            component_builder.create_component(component_name, cfg)
            for component_name in cfg.component_names
        ]
        return pipeline

    def persist(self,
                path: Text,
                persistor: Optional[Persistor] = None,
                project_name: Text = None,
                fixed_model_name: Text = None) -> Text:
        pass
