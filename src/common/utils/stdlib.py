import json
import datetime
from typing import Dict, Text

import six


def json_to_string(obj: Dict, **kwargs) -> str:
    indent = kwargs.pop("indent", 2)
    ensure_ascii = kwargs.pop("ensure_ascii", False)
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def override_defaults(defaults: Dict, custom: Dict) -> Dict:
    cfg = defaults or {}
    if custom:
        cfg.update(custom)
    return cfg


def get_tsp(fmt: Text = '%Y-%m-%d %H:%M:%S') -> Text:
    return datetime.datetime.now().strftime(fmt)


def module_path_from_object(o):
    """Returns the fully qualified class path of the instantiated object."""
    return o.__class__.__module__ + "." + o.__class__.__name__


def class_from_module_path(module_path):
    """Catch AttributeError and ImportError"""
    import importlib
    if "." in module_path:
        module_name, _, class_name = module_path.rpartition('.')
        m = importlib.import_module(module_name)
        return getattr(m, class_name)
    else:
        return globals()[module_path]


def ordered(obj):
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj


def as_text_type(t):
    if isinstance(t, six.text_type):
        return t
    else:
        return six.text_type(t)
