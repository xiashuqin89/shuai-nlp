from typing import Text, Tuple, Dict, SupportsFloat

import numpy as np


def preprocess(text: Text) -> Tuple:
    clean_text = text.lower().replace('.', ' .')
    words = clean_text.split(' ')
    word_to_id, id_to_word = {}, {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus: np.array,
                     vocab_size: int,
                     window_size: int = 1) -> np.array:
    """construct publicity matrix"""
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def cos_similarity(x: np.array, y: np.array, eps=1e-8) -> SupportsFloat:
    nx = x / (np.sqrt(np.sum(x ** 2))) + eps
    ny = y / (np.sqrt(np.sum(y ** 2))) + eps
    return np.dot(nx, ny)


def most_similar(query: Text,
                 word_to_id: Dict,
                 id_to_word: Dict,
                 word_matrix: np.array,
                 top=5):
    if query not in word_to_id:
        return None

    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(query_vec, word_matrix[i])

    return [
        f'{id_to_word[i]}:{similarity[i]}' for i in (-1 * similarity).argsort() if id_to_word[i] != query
    ]


def p_pmi(C: np.array, eps=1e-8):
    """calculate i,j relationship percent"""
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)
    return M


def test_cost_similarity(C: np.array, word1, word2):
    c0 = C[word_to_id[word1]]
    c1 = C[word_to_id[word2]]
    return cos_similarity(c0, c1)


if __name__ == '__main__':
    corpus, word_to_id, id_to_word = preprocess('You say goodbye and i say hello')
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)
    W = p_pmi(C)
    np.set_printoptions(precision=3)
    print(C)
    print(W)
