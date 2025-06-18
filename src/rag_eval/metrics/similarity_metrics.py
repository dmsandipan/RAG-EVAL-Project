import numpy as np
from numpy.linalg import norm

def jaccard_similarity(sentence_1,sentence_2):
    sentence_1 = set(sentence_1.split())
    sentence_2 = set(sentence_2.split())

    intesection = len(sentence_1.intersection(sentence_2))
    union = len(sentence_1.union(sentence_2))

    if union ==0:
        return 0
    return intesection/union

def cosine_similarity(A,B):
    return np.dot(A,B)/(norm(A)*norm(B))