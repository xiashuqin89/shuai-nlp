import copy
import os
from typing import Text, Optional, List, Dict, Any
from collections import defaultdict

from src.common import (
    TrainerModelConfig, TrainingData,
    logger, get_tsp, create_dir, module_path_from_object, make_path_absolute
)
from src.nlp import Persistor, ComponentBuilder, Metadata
from src.engine.constants import DEFAULT_PROJECT_NAME
from src.engine import Menu


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

        if component_builder is None:
            component_builder = ComponentBuilder()

        # if not self.skip_validation:
        #     validate_requirements(cfg.component_names)
        self.pipeline = self._build_pipeline(cfg, component_builder)

    def train(self, data: TrainingData, **kwargs) -> Dict[Text, Any]:
        """
        1, Trains the pipeline using the provided training data.
        2, checking all the input parameter (empty pipeline, pre layer component)
        3, every component run training model
        """
        self.training_data = data # domain.yml
        context = kwargs

        for component in self.pipeline:
            updates = component.provide_context()
            updates and context.update(updates)

        # if not self.skip_validation:
        #     validate_arguments(self.pipeline, context)

        working_data = copy.deepcopy(data)
        for i, component in enumerate(self.pipeline):
            logger.info(f"Starting to train component {component.name}")
            component.prepare_partial_processing(self.pipeline[:i], context)
            updates = component.train(working_data, self.config, **context)
            logger.info("Finished training component.")
            updates and context.update(updates)

        return context

    @staticmethod
    def _build_pipeline(cfg: TrainerModelConfig,
                        component_builder: ComponentBuilder) -> List:
        """Transform the passed names of the pipeline components into classes"""
        pipeline = [
            component_builder.create_component(component_name, cfg, Menu())
            for component_name in cfg.component_names
        ]
        return pipeline

    def persist(self,
                path: Text,
                persistor: Optional[Persistor] = None,
                project_name: Text = DEFAULT_PROJECT_NAME,
                fixed_model_name: Text = None) -> Text:
        """
        Persist all components of the pipeline to the passed path.
        1, generate storage path
        2, create dir
        3, write metadata to the json file
        todo mv save flow to storage module
        """
        metadata = defaultdict()
        metadata["language"] = self.config["language"]

        model_name = fixed_model_name if fixed_model_name else f'model_{get_tsp()}'
        path = make_path_absolute(path)
        dir_name = os.path.join(path, project_name, model_name)
        create_dir(dir_name)

        self.training_data and metadata.update(self.training_data.persist(dir_name))

        metadata["pipeline"] = []
        for component in self.pipeline:
            update = component.persist(dir_name)
            component_meta = component.component_config
            update and component_meta.update(update)
            component_meta["class"] = module_path_from_object(component)
            metadata["pipeline"].append(component_meta)

        Metadata(metadata, dir_name).persist(dir_name)

        if persistor is not None:
            persistor.persist(dir_name, model_name, project_name)
        logger.info(f'Successfully saved model into {os.path.abspath(dir_name)}')
        return dir_name
