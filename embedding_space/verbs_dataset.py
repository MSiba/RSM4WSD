import pandas as pd


path = "../data/wordnet_dataframes/VERBS.pickle"

verbs_df = pd.read_pickle(path)

# Transformations
# step 1. delete the virtual root "verb_root"
# step 2. calculate trans vector to put verb_root in the middle
# step 3. shift all cx with trans vector
# step 4. set y to y = -40.000 - 30.000
# this is to ensure that the families will be seperated from each other
root = 'verb_root'
initial_center_root = verbs_df.loc[verbs_df['synset']==root]['cx']
trans_vector = - initial_center_root

verbs_df['mod_start'] = verbs_df.apply(lambda row: row.start + trans_vector, axis=1)
verbs_df['mod_end'] = verbs_df.apply(lambda row: row.end + trans_vector, axis=1)
verbs_df["x"] = verbs_df.apply(lambda row: row.cx + trans_vector, axis=1)
# 40.000 is upper boundary of adjectives
# 30.000 is upper boundary of verbs
# Experiment: change those numbers to be more distant/nearer to each other
push_y = -40000-30000
verbs_df["y"] = push_y

#%%
pd.to_pickle(verbs_df, "../data/wordnet_dataframes/verbs_df.pickle")
