import pandas as pd


path = "../data/wordnet_dataframes/NOUNS.pickle"

nouns_df = pd.read_pickle(path)

# Transformations
# step 1. delete the virtual root "verb_root"
# step 2. calculate trans vector to put verb_root in the middle
# step 3. shift all cx with trans vector
# step 4. set y to y = -40.000 - 30.000
# this is to ensure that the families will be seperated from each other
root = 'entity.n.01'
initial_center_root = nouns_df.loc[nouns_df['synset']==root]['cx']
trans_vector = - initial_center_root

nouns_df['mod_start'] = nouns_df.apply(lambda row: row.start + trans_vector, axis=1)
nouns_df['mod_end'] = nouns_df.apply(lambda row: row.end + trans_vector, axis=1)
nouns_df["x"] = nouns_df.apply(lambda row: row.cx + trans_vector, axis=1)
# 40.000 is upper boundary of adjectives
# 165.000 is upper boundary of nouns
# Experiment: change those numbers to be more distant/nearer to each other
push_y = 165000 + 40000
nouns_df["y"] = push_y

#%%
pd.to_pickle(nouns_df, "../data/wordnet_dataframes/nouns_df.pickle")
