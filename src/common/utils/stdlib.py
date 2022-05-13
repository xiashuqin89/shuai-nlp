import json
import datetime
from typing import Dict, Text


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
