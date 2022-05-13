from typing import Text, Optional

from src.nlp.meta import Metadata
from src.nlp.components import Component
from src.common import TrainerModelConfig


class Menu(object):
    def load_component_by_name(self,
                               component_name: Text,
                               model_dir: Text,
                               metadata: Metadata,
                               cached_component: Optional[Component],
                               **kwargs) -> Optional[Component]:
        pass

    def create_component_by_name(self,
                                 component_name: Text,
                                 config: TrainerModelConfig) -> Optional[Component]:
        pass
