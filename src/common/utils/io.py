import io
import os
import json
import errno
from functools import wraps
from typing import Text, List, Any

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


def json_to_string(obj: Any, **kwargs):
    indent = kwargs.pop("indent", 2)
    ensure_ascii = kwargs.pop("ensure_ascii", False)
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def write_json_to_file(filename: Text, obj: Any, **kwargs):
    write_to_file(filename, json_to_string(obj, **kwargs))


def write_to_file(filename: Text, text: Text):
    with io.open(filename, 'w', encoding="utf-8") as f:
        f.write(str(text))


def create_dir(dir_path: Text):
    """Creates a directory and its super paths.
    Succeeds even if the path already exists."""
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
