import pandas as pd
from nltk.corpus import wordnet as wn
import numpy as np
from geometric_functions import find_angle



# TODO: run later
# # All words
# # 147.306
words = [word for word in wn.words()]
spatial_centers = []
for word in words:
    word_senses = [syn.name() for syn in wn.synsets(word)]
    word_information = {'word': word, 'senses': word_senses}
    spatial_centers.append(word_information)

"""
For centers: begin with random points (not centers!)
then see geometric center, barycenter, centroid, ...
https://en.wikipedia.org/wiki/Geometric_median
https://www.quora.com/What-is-barycenter-and-how-is-it-different-from-centroid
"""

def generate_center(df, center_info, fct="center"):
    direction_vec = np.zeros((1, 1))
    for center in center_info:

        N = len(center['senses'])
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)

        for i, sense in enumerate(center['senses']):
            x[i] = df.loc[df['synset']==sense]['x']
            y[i] = df.loc[df['synset'] == sense]['y']
            r[i] = df.loc[df['synset'] == sense]['radius'] / 2
            df.loc[df['synset'] == sense]["word"] = center["word"]

        if fct=="center":
            x_bar = (1/N) * np.sum(x)
            y_bar = (1/N) * np.sum(y)
            direction_vec[0] = x_bar
            direction_vec[1] = y_bar

        # TODO: include other functions that can be used for comparision
        # if fct == "centroid":
        # if fct == "center of mass"

        L0 = np.random.randint(low=1000, high=150000)
        WORD_POINT = direction_vec * L0 / np.linalg.norm(direction_vec)
        ALPHA = find_angle(np.array(1,0), direction_vec)

        center['word_point'] = WORD_POINT
        center['l0'] = L0
        center['alpha'] = ALPHA

    return center_info








