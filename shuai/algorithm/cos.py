from typing import List

import numpy as np


def cos(vec1: List, vec2: List) -> float:
    """cos similarity"""
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
