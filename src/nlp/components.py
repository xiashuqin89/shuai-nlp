from typing import Text

from src.common import TrainerModelConfig


class Component(object):
    pass


class ComponentBuilder(object):
    def create_component(self, component_name: Text, cfg: TrainerModelConfig) -> Component:
        pass
