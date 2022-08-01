from typing import Text

import Levenshtein


def levenshtein_distance(s1: Text, s2: Text) -> float:
    return Levenshtein.ratio(s1.upper(), s2.upper())
