"""
loading config
"""
from typing import Dict

from src.common.constants import DEFAULT_CONFIG
from src.common.utils.stdlib import json_to_string, override_defaults
from src.common.log import logger
from src.common.exceptions import InvalidConfigError


def override_defaults(defaults: Dict, custom: Dict) -> Dict:
    cfg = defaults or {}
    if custom:
        cfg.update(custom)
    return cfg


class TrainerModelConfig(object):
    """
    transfer user's yaml or json config
    to python object
    """

    def __init__(self, configuration_values=None):

        if not configuration_values:
            configuration_values = {}

        self.override(DEFAULT_CONFIG)
        self.override(configuration_values)

        # user use a native pipeline to replace config
        # if isinstance(self.__dict__['pipeline'], str):
        #     from rasa_nlu import registry
        #
        #     template_name = self.__dict__['pipeline']
        #     pipeline = registry.pipeline_template(template_name)
        #
        #     if pipeline:
        #         # replaces the template with the actual components
        #         self.__dict__['pipeline'] = pipeline
        #     else:
        #
        #         raise InvalidConfigError("No pipeline specified and unknown ")

        for key, value in self.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def __getstate__(self):
        return self.as_dict()

    def __setstate__(self, state):
        self.override(state)

    def items(self):
        return list(self.__dict__.items())

    def as_dict(self):
        return dict(list(self.items()))

    def view(self):
        return json_to_string(self.__dict__, indent=4)

    def for_component(self, name, defaults=None):
        for c in self.pipeline:
            if c.get("name") == name:
                return override_defaults(defaults, c)
        else:
            return defaults or {}

    @property
    def component_names(self):
        if self.pipeline:
            return [c.get("name") for c in self.pipeline]
        else:
            return []

    def set_component_attr(self, name, **kwargs):
        for c in self.pipeline:
            if c.get("name") == name:
                c.update(kwargs)
        else:
            logger.warn("Tried to set configuration value for component '{}' "
                        "which is not part of the pipeline.".format(name))

    def override(self, config):
        if config:
            self.__dict__.update(config)
