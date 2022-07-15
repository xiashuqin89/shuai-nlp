from typing import Text, Optional, Dict, Type

from shuai.nlp.meta import Metadata
from shuai.nlp.components import Component
from shuai.common import TrainerModelConfig, class_from_module_path
from .default_components import COMPONENT_CLASSES


class InvalidRecipeException(Exception):
    def __init__(self, message: Text):
        self.message = message

    def __str__(self):
        return self.message


class Menu(object):
    """
    Use all the native component register
    This class like a db, native component all from this
    todo should be replace by decorator & class name maybe update
    """
    _registered_components: Dict[Text, Optional[Type[Component]]] = {c.name: c for c in COMPONENT_CLASSES}

    def get_component_class(self, component_name: Text) -> Optional[Type[Component]]:
        if component_name not in self._registered_components:
            try:
                return class_from_module_path(component_name)
            except InvalidRecipeException:
                raise InvalidRecipeException(
                    f"Failed to find component class for '{component_name}'. Unknown ")
        return self._registered_components[component_name]

    def load_component_by_name(self,
                               component_name: Text,
                               model_dir: Text,
                               metadata: Metadata,
                               cached_component: Optional[Component],
                               **kwargs) -> Optional[Component]:
        component_clz = self.get_component_class(component_name)
        return component_clz.load(model_dir, metadata, cached_component, **kwargs)

    def create_component_by_name(self,
                                 component_name: Text,
                                 config: TrainerModelConfig) -> Optional[Component]:
        component_clz = self.get_component_class(component_name)
        return component_clz.create(config)
