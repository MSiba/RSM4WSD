import pandas as pd


path = "../data/wordnet_dataframes/ADJECTIVES.pickle"

adjectives_df = pd.read_pickle(path)

# Transformations
# step 1. get the virtual root "adjective_root"
# step 2. calculate trans vector to put adjective_root in the middle
# step 3. shift all y with trans vector
# step 4. set x to x = 150.0000
# this is to ensure that the families will be seperated from each other
root = 'adjective_root'
initial_center_root = adjectives_df.loc[adjectives_df['synset']==root]['cx']
trans_vector = - initial_center_root

adjectives_df['mod_start'] = adjectives_df.apply(lambda row: row.start + trans_vector, axis=1)
adjectives_df['mod_end'] = adjectives_df.apply(lambda row: row.end + trans_vector, axis=1)
adjectives_df["y"] = adjectives_df.apply(lambda row: row.cx + trans_vector, axis=1)
# push the x-coordinates 150.000 unit to the right(+)
# Experiment: change those numbers to be more distant/nearer to each other
push_x = 150000
adjectives_df["x"] = push_x

#%%
pd.to_pickle(adjectives_df, "../data/wordnet_dataframes/adjectives_df.pickle")
