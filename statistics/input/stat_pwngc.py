import pandas as pd
from matplotlib import pyplot as plt
#%%
path = "C:/Users/HP/PycharmProjects/RSM4WSD/data/wordnet_dataframes/SPATIAL_WORDNET.pickle"
params = ["l0", "alpha", "l_i", "beta_i", "radius"]

original_spatial_wordnet = pd.read_pickle(path)
myFig = plt.figure()
original_statistics = original_spatial_wordnet[params].astype(float).describe(include='all')
original_boxplot = original_spatial_wordnet[params].astype(float).boxplot(column=params)
myFig.savefig("spatial_wordnet_params_ordofmag.svg", format="svg")