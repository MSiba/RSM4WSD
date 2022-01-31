import pandas as pd

df = pd.read_pickle("../data/wordnet_dataframes/SPATIAL_WORDNET.pickle")

def prune(df):
    """
    The current SPATIAL_WORDNET.pickle dataset contains for the same synset, several words, and for each word, several parameters.
    Example: for the synset "dog.n.01", there are 3 entries in the data
            111113  canis_familiaris  dog.n.01  ...  133679.594621     NaN
            111114               dog  dog.n.01  ...  104074.378875    9.99
            111115      domestic_dog  dog.n.01  ...   90470.594621     0.0

    Problem: how to tag the training corpora with this dataset? It is confusing/misleading.
            Also, the results will not be accurate because the algorithm is not deterministic about the sense!
            I decided: the word column is not so important? as synset column (for the task of WSD at lease), because
                       the input words are usual words, e.g. 'dog', not 'domestic_dog'. In addition,
                       the polysemous word, e.g. dog is more important than domestic_dog/canis familiaris,
                       because it will not probably not appear in the data.
    Solution: If the synset appears several times in the training set with different words, keep only the polysemous one.

    OR: if I don't want to get remove words, I can simply override the values, based on the most polysemouos word
    :param df:
    :return:
    """
    virtual_roots = ["verb_root", "adjective_root", "adverb_root"]
    synsets = set(df["synset"]) - set(virtual_roots)
    for syn in synsets:
        syn_df = df.loc[df["synset"]==syn]
        if syn_df.shape[0] <= 1:
            continue
        # else:




    return df