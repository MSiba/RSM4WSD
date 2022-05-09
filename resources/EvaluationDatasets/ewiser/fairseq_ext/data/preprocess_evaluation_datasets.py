import os
import itertools
import pandas as pd
import numpy as np
import torch
import statistics
from collections import defaultdict


from read_evaluation_datasets import read_gold, read_xml_from_Raganato, extract_full_sent
from resources.spatial_tagging_utils import spatial_tag


"__author__ == Siba Mohsen"

# Define all variables
# testing dataset's name

# NAME = "senseval2"
NAMES = ["senseval2", "senseval3", "semeval2007", "semeval2013", "semeval2015", "ALL"]

# paths
path = "C:/Users/HP/PycharmProjects/RSM4WSD/resources/EvaluationDatasets/eval_datasets/"
testing_path = "C:/Users/HP/PycharmProjects/RSM4WSD/data/testing_datasets/"


def preprocess_testing_data(name):
    """
    Reads the xml file using Raganato's parser in ewiser. Preprocesses the extracted information as
    id  |   lemma   |   pos |   sense(lemma_from_key)   |   spatial_params_id   |   spatial_params
    xml |   xml     |   xml |   key.txt                 |   spatial_dataframe   |   spatial_dataframe
    :param name: str. The name of the testing dataset
    :return: dictionaries of id, lemma, pos, sense, and spatial parameters for each sentence, stored in data/testing_datasets
    """

    # paths
    # path = "C:/Users/HP/PycharmProjects/RSM4WSD/resources/EvaluationDatasets/eval_datasets/"
    dataset_xml = name + "/" + name + ".data.xml"
    #"senseval2/senseval2.data.xml"
    dataset_gold = name + "/" + name + ".gold.key.txt"
    #"senseval2/senseval2.gold.key.txt"

    # dictionaries
    spatial_ids = defaultdict(list)
    spatial_tags = defaultdict(list)
    spatial_df = pd.read_pickle("C:/Users/HP/PycharmProjects/RSM4WSD/data/wordnet_dataframes/SPATIAL_WORDNET.pickle")

    # save testing dataset
    # save with torch such that the index and the spatial labels stay in torch type
    # it is better because when saving them to pkl, torch vanishes
    # testing_path = "C:/Users/HP/PycharmProjects/RSM4WSD/data/testing_datasets/"
    testing_data_file = testing_path + name + ".pt"
    #"senseval2.pt"


    # read the gold dataset
    t, wnt = read_gold(gold_path=path+dataset_gold)

    # read the .xml file using Raganato's xml parser, and extract structured information
    # id, lemma, pos, sense(lemma_from_key), spatial_params
    ids, lemmas, pos, senses = read_xml_from_Raganato(xml_path=path+dataset_xml, wnt=wnt)


    sentences = []
    for (k1, v1), (k2, v2), (k3, v3), (k4, v4) in zip(ids.items(),
                                                     lemmas.items(),
                                                     pos.items(),
                                                     senses.items()):
        tmp_sentence = []
        pos2wn = {"NOUN": "n",  # noun
                  "ADJ": "a",  # adjective
                  "VERB": "v",  # verb
                  "ADV": "r"}  # adverb

        for i in range(len(v1)):
            word, lemma, pos_tag, sense = v1[i], v2[i], v3[i], v4[i]
            # print(word, lemma, pos_tag, sense)
            indices = []
            tags = []
            for wn_syn in sense:
                idx, tag = spatial_tag(df=spatial_df, word=lemma, synset=wn_syn)
                indices.append(idx)
                tags.append(tag)

            tmp_sentence.append([word, lemma, pos2wn[pos_tag], sense, torch.tensor(indices, dtype=torch.int64), torch.tensor(tags, dtype=torch.float32)])
        sentences.append(tmp_sentence)

        # print(sentences)

    torch.save(sentences, testing_data_file)

    print("Some statistics about the testing dataset {}.".format(name))
    print()
    print("Number of sentences: ", len(sentences))
    print()
    nb_words = [len(v) for k, v in ids.items()]
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


def store_full_sent(xml_name):
    sentences = extract_full_sent(os.path.join(path, xml_name, xml_name + ".data.xml"))
    torch.save(sentences, os.path.join(testing_path, "full_sentences", 'sent_' + xml_name + '.pt'))

#%%
# tes = torch.load(senseval2_file)
#%%
# for testing_name in NAMES:
#     preprocess_testing_data(testing_name)

#%%
preprocess_testing_data('senseval3')
#%%
for testing_name in NAMES:
    store_full_sent(testing_name)
