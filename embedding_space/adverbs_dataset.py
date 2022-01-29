import pandas as pd


path = "../data/wordnet_dataframes/ADVERBS.pickle"

adverbs_df = pd.read_pickle(path)

# Transformations
# step 1. get the virtual root "adverb_root"
# step 2. calculate trans vector to put adverb_root in the middle
# step 3. shift all y with trans vector
# step 4. set x to x = -150.0000
# this is to ensure that the families will be seperated from each other
root = 'adverb_root'
initial_center_root = adverbs_df.loc[adverbs_df['synset']==root]['cx']
trans_vector = - initial_center_root

adverbs_df['mod_start'] = adverbs_df.apply(lambda row: row.start + trans_vector, axis=1)
adverbs_df['mod_end'] = adverbs_df.apply(lambda row: row.end + trans_vector, axis=1)
adverbs_df["y"] = adverbs_df.apply(lambda row: row.cx + trans_vector, axis=1)
# push the x-coordinates 150.000 unit to the right(+)
# Experiment: change those numbers to be more distant/nearer to each other
push_x = -150000
adverbs_df["x"] = push_x

#%%
pd.to_pickle(adverbs_df, "../data/wordnet_dataframes/adverbs_df.pickle")
