import pandas as pd

src = "../data/wordnet_dataframes/"

nouns_df = pd.read_pickle(src + "nouns_df.pickle")
verbs_df = pd.read_pickle(src + "verbs_df.pickle")
adjectives_df = pd.read_pickle(src + "adjectives_df.pickle")
adverbs_df = pd.read_pickle(src + "adverbs_df.pickle")

dataframes = [nouns_df, verbs_df, adjectives_df, adverbs_df]

spatial_wordnet_df = pd.concat(dataframes)

#%%
spatial_wordnet_df.to_pickle(src + "spatial_wordnet_df.pickle")