import pandas as pd
import numpy as np
import numpy as np
from helper_functions import find_angle

src = "../data/wordnet_dataframes/"
words_df = pd.read_pickle(src + "exploded_word_mapping_POS.pickle")
senses_df = pd.read_pickle(src + "spatial_wordnet_df.pickle")

#%%
# join by sense
# are there entries, where the sense is unidentified?
# change the name of senses in word_df to synset
words_df = words_df.rename(columns={"senses": "synset"})
#%%
# 227.736 (including virtual roots)
word_sense_df = pd.merge(words_df, senses_df, on='synset', how='outer')

#%%
# reduce the radius
word_sense_df["radius"] = word_sense_df["radius"]/2

#%%

#%%
# add to this dataframe li, and beta
word_sense_df["l0"] = word_sense_df.apply(lambda row: np.round(row.l0, 2), axis=1)
word_sense_df["x"] = word_sense_df.apply(lambda row: np.round(row.x, 2), axis=1)
word_sense_df["y"] = word_sense_df.apply(lambda row: np.round(row.y, 2), axis=1)
word_sense_df["word_point"] = word_sense_df.apply(lambda row: np.around(row.word_point, decimals=2), axis=1)
word_sense_df["alpha"] = word_sense_df.apply(lambda row: np.round(row.alpha, 2), axis=1)
word_sense_df["l_i"] = word_sense_df.apply(lambda row: np.round(np.linalg.norm(
                                        np.array([row.x, row.y]) - row.word_point), 2), axis=1)
word_sense_df["beta_i"] = word_sense_df.apply(lambda row: np.round(find_angle(row.word_point,
                                                                              np.array([row.x, row.y]) - row.word_point),
                                                                   2), axis=1)

#%%
# do not remove "verb_root", "adjective_root", "adverb_root", because they can help us later on
# indices of these words stored in synsets
virtual_roots = ["verb_root", "adjective_root", "adverb_root"]
for root in virtual_roots:
    word_sense_df.loc[word_sense_df['synset'] == root,
                      ['word', 'word_point', 'l0', 'alpha', 'pos', 'l_i', 'beta_i']] = \
        [np.array([root, np.array([0,0], dtype=object), 0.0, 0.0, 'virtual_root', 0.0, 0.0], dtype=object)]

#%%
word_sense_df.to_pickle("../data/wordnet_dataframes/SPATIAL_WORDNET.pickle")