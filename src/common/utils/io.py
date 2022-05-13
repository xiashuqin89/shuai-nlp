import io
import os
from functools import wraps
from typing import Text, List

import yaml


def fix_yaml_loader(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        from yaml import Loader, SafeLoader

        def construct_yaml_str(self, node):
            # Override the default string handling function
            # to always return unicode objects
            return self.construct_scalar(node)

        Loader.add_constructor(u'tag:yaml.org,2002:str', construct_yaml_str)
        SafeLoader.add_constructor(u'tag:yaml.org,2002:str', construct_yaml_str)
        return func(*args, **kwargs)
    return _wrapper


@fix_yaml_loader
def read_yaml_file(filename):
    return yaml.load(read_file(filename, "utf-8"))


def read_file(filename, encoding="utf-8"):
    """Read text from a file."""
    with io.open(filename, encoding=encoding) as f:
        return f.read()


def list_files(path: Text) -> List[Text]:
    pass
