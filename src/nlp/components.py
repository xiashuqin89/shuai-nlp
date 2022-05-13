from typing import Text, List, Dict, Optional, Any

from src.common import TrainerModelConfig, TrainingData


class Component(object):
    @classmethod
    def required_packages(cls) -> List[Text]:
        return []

    def provide_context(self) -> Optional[Dict[Text, Any]]:
        pass

    def prepare_partial_processing(self, pipeline: List[object], context: Dict):
        self.partial_processing_pipeline = pipeline
        self.partial_processing_context = context

    def train(self, training_data: TrainingData, cfg: TrainerModelConfig, **kwargs):
        """Train this component.
        This is the components chance to train itself provided
        with the training data. The component can rely on
        any context attribute to be present, that gets created
        by a call to `pipeline_init` of ANY component and
        on any context attributes created by a call to `train`
        of components previous to this one."""
        pass

    def persist(self, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading."""
        pass


class ComponentBuilder(object):
    def create_component(self, component_name: Text, cfg: TrainerModelConfig) -> Component:
        pass


def validate_requirements(component_names: List[Text]):
    """Ensures that all required python packages are installed to
        instantiate and used the passed components."""
    pass
