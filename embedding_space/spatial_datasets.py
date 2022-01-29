import pandas as pd
from nltk.corpus import wordnet as wn
import numpy as np
from geometric_functions import find_angle
#%%

# # All words
# # 147.306
words = [word for word in wn.words()]
spatial_centers = []
for word in words:
    word_senses = [syn.name() for syn in wn.synsets(word)]
    word_information = {'word': word, 'senses': word_senses}
    spatial_centers.append(word_information)

#%%
"""
For centers: begin with random points (not centers!)
then see geometric center, barycenter, centroid, ...
https://en.wikipedia.org/wiki/Geometric_median
https://www.quora.com/What-is-barycenter-and-how-is-it-different-from-centroid
"""

def generate_center(df, center_info, fct="center"):
    direction_vec = np.zeros((2))
    for center in center_info:
        print("Processing the word {}".format(center['word']))

        N = len(center['senses'])
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)

        for i, sense in enumerate(center['senses']):
            print("Processing the sense {}".format(sense))
            x[i] = df.loc[df['synset']==sense]['x']
            y[i] = df.loc[df['synset'] == sense]['y']
            r[i] = df.loc[df['synset'] == sense]['radius'] / 2

        if fct == "center":
            x_bar = (1/N) * np.sum(x)
            y_bar = (1/N) * np.sum(y)
            direction_vec[0] = x_bar
            direction_vec[1] = y_bar

        # TODO: include other functions that can be used for comparision
        # if fct == "centroid":
        # if fct == "center of mass"

        L0 = np.random.randint(low=1000, high=150000)
        WORD_POINT = direction_vec * L0 / np.linalg.norm(direction_vec)
        ALPHA = find_angle(np.array([1, 0]), direction_vec)

        center['word_point'] = WORD_POINT
        center['l0'] = L0
        center['alpha'] = ALPHA

    return center_info

#%%
# loads the spatial wordnet, to map the static word
spatial_wordnet_df = pd.read_pickle("../data/wordnet_dataframes/spatial_wordnet_df.pickle")
spatial_wordnet_df["word"] = ""
#%%
# Number of all wordnet words is 147.306 WORDS, their synsets count is 117.659 synsets
# In spatial_centers, the number of synsets is 227.733 synsets. This means that the majority of the synsets have two
# center words, the static one (e.g. flower), and the most specific synset (e.g. red_flower)
# Number of all spatial_wordnet_df is 117.662 SYNSETS(instead of 117.597, because I add roots for verbs, adverbs, and adjectives
# when I encode the hierarchy of the tree using DFS algorithm)
# If I look into the words,
# difference = set(words) - set(spatial_wordnet_df["synset"])
# I can simply conclude that those words were mentioned in Wordnet but are linked to no synset
#%%
static_words = generate_center(spatial_wordnet_df, center_info=spatial_centers)

#%%
# transform into dataframe
static_words_df = pd.DataFrame.from_records(static_words)
# #%%
# static_words_df.to_pickle("../data/wordnet_dataframes/word_mapping_df.pickle")
#%%
# flatten the senses for each word as a single column
# initial size of static_words_df is 147.306
# After falttening, size is: 227.733
column2explode = "senses"
# reset index to assign each entry a unique index
explode_static_words_df = static_words_df.explode(column2explode).reset_index(drop=True)

#%%
# save exploded dataframe into pickle file
# explode_static_words_df.to_pickle("../data/wordnet_dataframes/exploded_word_mapping.pickle")
#%%
def get_wn_pos(name):
    synset = wn.synset(name)
    pos = synset.pos()
    if pos == 's':
        pos = 'a'
    return pos

explode_static_words_df["pos"] = explode_static_words_df.apply(lambda row: get_wn_pos(row.senses), axis=1)
#%%
# explode_static_words_df.to_pickle("../data/wordnet_dataframes/exploded_word_mapping_POS.pickle")