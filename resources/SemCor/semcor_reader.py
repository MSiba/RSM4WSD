import string
from collections import defaultdict
import itertools
import statistics

import numpy as np
import pandas as pd
import torch

import nltk
nltk.download('semcor')
from nltk.corpus import semcor

from resources.PWNGC.my_classes import treebank_tagset, treebank2wordnet
from spatial_tagging_utils import spatial_tag



# replace the path to SPATIAL_WORDNET
SPATIAL_WORDNET_PICKLE = "C:/Users/HP/PycharmProjects/RSM4WSD/data/wordnet_dataframes/SPATIAL_WORDNET.pickle"

def read_semcor():
    print("Reading")
    """
    Reads the semcor tagged sentences with variable 'both'. Organizes the extracted information as
    lemma   |   pos |   sense(lemma_from_key)   |   spatial_params

    :return: dictionaries of lemma, pos, sense, and spatial parameters for each sentence
    """
    # get sentences from Semcor as tokenized lists
    # 37176 sentences in total
    # semcor_sentences = semcor.sents()

    # get the tagged senses for each sentence
    # tag is set to "both" to get POS + semantic tagging
    semcor_tags = semcor.tagged_sents(tag='both')

    print("Imported semcor tags")

    # initialize a defaultdict to group words of same sentence
    # {"0": [element1, ]}
    sentences = defaultdict(list)

    for i, sent in enumerate(semcor_tags):
        sentences[i] = sent

    lemmas = {k: list(
        map((lambda x: x.leaves()[0].lower() if len(x.leaves()) == 1 else "_".join(list(map(str.lower, x.leaves())))),
            lst)) for k, lst in sentences.items()}

    print("Lemmas' dictionary is done")

    pos_tags = {k: list(map(
        lambda x: treebank2wordnet(x.pos()[0][1]) if x.pos()[0][1] in treebank_tagset and treebank2wordnet(
            x.pos()[0][1]) != '' else 'None', lst)) for k, lst in sentences.items()}

    print("POS Tags' dictionary is done")

    senses = {k: list(map(
        lambda x: x.label().synset().name() if (not isinstance(x.label(), str)) and (x.label() is not None) else 'None',
        lst)) for k, lst in sentences.items()}

    print("Senses' dictionary is done")



    # None indices
    none_ids = {k: np.where(np.array(val)=='None')[0].tolist() for k, val in senses.items()}

    print("extracted none indices")

    # delete None indices
    valid_lemmas = lemmas
    valid_pos_tags = pos_tags
    valid_senses = senses
    for (k1, val1), (k2, val2), (k3, val3) in zip(valid_lemmas.items(), valid_pos_tags.items(), valid_senses.items()):
        for i in reversed(none_ids[k1]):
            del val1[i]
            del val2[i]
            del val3[i]

   # check if there are sentences having two different possibilies for senses


    # ###############################################################################
    # Do some checks before writing to testing files
    # ###############################################################################
    print("Checking if all sentences have the same number of words and tags")
    count_lemmas = {k: len(val) for k, val in lemmas.items()}
    count_pos = {k: len(val) for k, val in pos_tags.items()}
    count_senses = {k: len(val) for k, val in senses.items()}

    N = len(count_senses.keys())
    for i in range(N):
       assert count_lemmas[i] == count_pos[i] == count_senses[i], "The number of elements is" \
                                                                                  " inconsistent for key {}".format(i)

    assert len(valid_lemmas) == len(valid_pos_tags) == len(senses), "Detected inconsistency while " \
                                                                                      "reading the testing data " \
                                                                                      "len(valid_lemmas) = {} len(valid_pos_tags) = {} len(senses) = {}.".format(
        len(valid_lemmas), len(valid_pos_tags), len(senses))

    return valid_lemmas, valid_pos_tags, senses

#%%

def preprocess_semcor(lemmas, pos, senses):
    print("Start Preprocessing")


    spatial_df = pd.read_pickle(SPATIAL_WORDNET_PICKLE)

    # save testing dataset
    # save with torch such that the index and the spatial labels stay in torch type
    # it is better because when saving them to pkl, torch vanishes
    training_path = "C:/Users/HP/PycharmProjects/RSM4WSD/data/training_datasets/"
    training_semcor_file = training_path + "idx_complete_semcor4regressor" + ".pt"

    sentences = []

    print("Spatial Tagging")

    for (k1, v1), (k2, v2), (k3, v3) in zip(lemmas.items(),
                                            pos.items(),
                                            senses.items()):
        tmp_sentence = []

        for i in range(len(v1)):
            lemma, pos_tag, sense = v1[i], v2[i], v3[i]
            print(lemma, pos_tag, sense)
            idx, tag = spatial_tag(df=spatial_df, word=lemma, synset=sense)
            tmp_sentence.append([lemma, pos_tag, sense, np.array(idx, dtype=np.int64),
                                 np.array(tag, dtype=np.float64)])


        sentences.append(tmp_sentence)

        # print(sentences)

    torch.save(sentences, training_semcor_file)
    print("SemCor saved in torch")
    print()
    print("Some statistics about the training dataset SemCor.")
    print()
    print("Number of sentences: ", len(sentences))
    print()
    nb_words = [len(v) for k, v in lemmas.items()]
    words_nb, sentence_nb = np.unique(nb_words, return_counts=True)
    print("Number of words per sentence: ")
    for wnb, snb in zip(words_nb, sentence_nb):
        print(wnb, ": ", snb)
    print()
    avg_words = statistics.mean(nb_words)
    print("Average words per sentence: ", avg_words)
    print()
    flatten_pos = list(itertools.chain(*list(pos.values())))
    labels, counts = np.unique(flatten_pos, return_counts=True)
    print("Available POS tags: ")
    for l, c in zip(labels, counts):
        print(l, ": ", c)
    print("Total : ", len(flatten_pos))

#%%
lemmas, pos, senses = read_semcor()

#%%
preprocessing = preprocess_semcor(lemmas, pos, senses)