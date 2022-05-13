import json
from typing import Dict


def json_to_string(obj: Dict, **kwargs) -> str:
    indent = kwargs.pop("indent", 2)
    ensure_ascii = kwargs.pop("ensure_ascii", False)
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def override_defaults(defaults: Dict, custom: Dict) -> Dict:
    cfg = defaults or {}
    if custom:
        cfg.update(custom)
    return cfg
