"__author__ == Siba Mohsen"


def spatial_tag(df, word, synset):
    """
    Finds the spatial tag and index of the a synset based on the word lemma.
    :param df: pandas dataframe. The dataframe storing the wordnet words, synsets and their spatial tags
    :param word: string. The sense-annotated word.
    :param synset: string. The initial WordNet annotation of the word
    :return: index and spatial tag of the word and synset.
    """
    # print("Word = {} ; Synset = {}".format(word, synset))

    if synset == "no-synset":
        return "O"

    word = word.lower()  # .replace("'", "")
    #print(word)

    params = ["l0", "alpha", "l_i", "beta_i", "radius"]
    tag = None
    syn_df = df.loc[df["synset"] == synset]
    #print(syn_df)
    # try:
    if syn_df.shape[0] == 1:
        print("The word has only 1 synset :)")
        tag = syn_df[params].to_numpy()[0]

        idx = syn_df.index[0]
        # print("idx = ", idx, type(idx))

        #print(" There is only one match")
        #print("tag = {}".format(tag))
        return idx, tag

    elif syn_df.shape[0] > 1:
        print("try to find best match based on synset tag + word")
        # print("syn_df", syn_df)
        try:
            # try to find best match based on synset tag + word
            best_match = syn_df.loc[syn_df["word"] == word]

            idx = best_match.index[0]
            #print("best match = ", best_match)
            #print("idx = ", idx, type(idx))
            tag = best_match[params].to_numpy()[0]
            #print("best match = {}".format(best_match))
            #print("tag = {}".format(tag))
            return idx, tag
        except:
            print("there is no best match, take any value of the subset synset")
            # if there is no best match, take any value of the subset synset
            random_match = syn_df.sample()
            idx = random_match.index[0]
            tag = random_match[params].to_numpy()[0]
            #print("there is no best match, take any value of the subset synset")
            ##print("random match = {}".format(random_match))
            # print("idx = ", idx, type(idx))
            # print("tag = {}".format(tag))
            return idx, tag

