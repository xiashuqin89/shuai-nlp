from typing import Text

import Levenshtein


def core_method(s1: Text, s2: Text) -> float:
    return Levenshtein.ratio(s1.upper(), s2.upper())
