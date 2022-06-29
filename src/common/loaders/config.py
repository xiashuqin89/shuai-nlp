"""
loading config
"""
import os
from typing import Dict

import yaml

from src.common.constants import (
    DEFAULT_CONFIG, DEFAULT_CONFIG_LOCATION,
    REGISTERED_PIPELINE_TEMPLATES
)
from src.common.utils.stdlib import json_to_string, override_defaults
from src.common.utils.io import read_yaml_file
from src.common.log import logger
from src.common.exceptions import InvalidConfigError


def load(filename=None, **kwargs):
    if filename is None and os.path.isfile(DEFAULT_CONFIG_LOCATION):
        filename = DEFAULT_CONFIG_LOCATION

    if filename is not None:
        try:
            file_config = read_yaml_file(filename)
        except yaml.parser.ParserError as e:
            raise InvalidConfigError("Failed to read configuration file "
                                     "'{}'. Error: {}".format(filename, e))
        if kwargs:
            file_config.update(kwargs)
        return TrainerModelConfig(file_config)
    else:
        return TrainerModelConfig(kwargs)


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

        if isinstance(self.__dict__['pipeline'], str):
            template_name = self.__dict__['pipeline']
            components = REGISTERED_PIPELINE_TEMPLATES.get(template_name)
            if components:
                # replaces the template with the actual components
                self.__dict__['pipeline'] = [{"name": c} for c in components]
            else:
                raise InvalidConfigError("No pipeline specified and unknown ")

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
