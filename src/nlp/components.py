from typing import Text, List, Dict, Optional, Any, Tuple

from src.common import (
    TrainerModelConfig, TrainingData,
    override_defaults, logger,
    MissingArgumentError,
)
from src.nlp.meta import Metadata
from src.engine import Menu


class Component(object):
    name = ""
    provides = []
    requires = []
    defaults = {}

    language_list = None

    def __init__(self, component_config: Dict[Text, Any] = None):
        self.component_config = override_defaults(
                self.defaults, component_config)

        self.partial_processing_pipeline = None
        self.partial_processing_context = None

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
    """
    Creates trainers based on configurations.
    1, Try to load from cache
    2, If not cache, register it
    """

    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.component_cache = {}

    def _add_to_cache(self, component: Component, cache_key: Text):
        if cache_key is not None and self.use_cache:
            self.component_cache[cache_key] = component
            logger.info(f'Added "{component.name}" to component cache. Key "{cache_key}".')

    def _get_from_cache(self,
                        component_name: Text,
                        model_metadata: Metadata,
                        menu: Menu) -> Tuple[Optional[Component], Optional[Text]]:

        component_class = menu.get_component_class(component_name)
        cache_key = component_class.cache_key(model_metadata)
        if cache_key is not None and self.use_cache and cache_key in self.component_cache:
            return self.component_cache[cache_key], cache_key
        return None, cache_key

    def load_component(self,
                       component_name: Text,
                       model_dir: Text,
                       model_metadata: Metadata,
                       menu: Menu,
                       **context) -> Component:
        """Tries to retrieve a component from the cache, calls
        `load` to create a new component."""

        try:
            cached_component, cache_key = self._get_from_cache(
                    component_name, model_metadata, menu)
            component = menu.load_component_by_name(
                    component_name, model_dir, model_metadata,
                    cached_component, **context)
            if not cached_component:
                self._add_to_cache(component, cache_key)
            return component
        except MissingArgumentError as e:
            logger.error(f'Failed to load component "{component_name}". {e}')
            raise

    def create_component(self,
                         component_name: Text,
                         cfg: TrainerModelConfig,
                         menu: Menu) -> Component:
        try:
            component, cache_key = self._get_from_cache(
                component_name, Metadata(cfg.as_dict(), None))
            if component is None:
                component = menu.create_component_by_name(component_name, cfg)
                self._add_to_cache(component, cache_key)
            return component
        except MissingArgumentError as e:  # pragma: no cover
            logger.error(f'Failed to load component "{component_name}". {e}')
            raise


def validate_requirements(component_names: List[Text]):
    """Ensures that all required python packages are installed to
        instantiate and used the passed components."""
    pass
