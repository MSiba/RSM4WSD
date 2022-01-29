import pandas as pd
import numpy as np

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
import numpy as np
from geometric_functions import find_angle
# add to this dataframe li, and beta
word_sense_df["l_i"] = word_sense_df.apply(lambda row: np.linalg.norm(np.array([row.x, row.y]) - row.word_point), axis=1)
word_sense_df["beta_i"] = word_sense_df.apply(lambda row: np.round(find_angle(row.word_point, np.array([row.x, row.y])), 2), axis=1)

#%%
# do not remove "verb_root", "adjective_root", "adverb_root", because they can help us later on
# indices of these words stored in synsets
virtual_roots = ["verb_root", "adjective_root", "adverb_root"]
for root in virtual_roots:
    word_sense_df.loc[word_sense_df['synset'] == root, ['word', 'word_point', 'l0', 'alpha', 'pos', 'l_i', 'beta_i']] = [[root, np.array([0,0]), 0, 0, 'virtual_root', 0, 0]]

#%%
word_sense_df.to_pickle("../data/wordnet_dataframes/SPATIAL_WORDNET.pickle")