import math
import numpy as np
from nltk.corpus import wordnet as wn



def RELATION(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    relational_matrix = np.zeros((len(synsets1), len(synsets2)))
    for i in synsets1:
        for j in synsets2:
            hyper_1 = set.intersection(set())


def find_angle(vec1, vec2):
    unit_1 = vec1/np.linalg.norm(vec1)
    unit_2 = vec2/np.linalg.norm(vec2)
    angle = np.arccos(np.dot(unit_1, unit_2))
    return angle * 180.0 / np.pi # in degrees

def translate(point, dist): #acts as prolongement
    return

def rotate(center, point, alpha):
    return

def enlarge(center, radius, length):
    return

