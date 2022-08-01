import numpy as np


def cos(vec1: np.array, vec2: np.array) -> float:
    """cos similarity"""
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
