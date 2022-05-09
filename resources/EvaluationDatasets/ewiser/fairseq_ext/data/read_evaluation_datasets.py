from collections import defaultdict
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
import torch

from ewiser.fairseq_ext.data.wsd_dataset import *


"__author__ = Siba Mohsen"


def read_gold(gold_path):
    """
    Reads the sense annotations in the gold file.
    Example of the gold content:
    d000.s009.t001 today%1:28:01::
        d000: document 0
        s009: sentence 9
        t001: the word's position in the sentence is 2
        today%1:28:01:: is the wordnet offset for the word "today"

    :param gold_path: string. Path to the gold annotations
    :return: dictionaries of wordnet tags. {sentence_number: [wn_tag1, wn_tag2, ...], ...}
    """
    tags = defaultdict(list)
    wn_tags = defaultdict(list)
    with open(gold_path, 'r') as read_tags:
        lines = read_tags.readlines()

        for line in lines:
            id_tag = line[:-1].split(' ')
            # print(id_tag)
            tags[id_tag[0]].extend(id_tag[1:])
            wn_tags[id_tag[0]].extend(list(map(lambda x: wn.lemma_from_key(x).synset().name(), id_tag[1:])))

        read_tags.close()

    return tags, wn_tags


def read_xml_from_Raganato(xml_path, wnt):
    """
    Reads the xml file using Raganato's parser in ewiser. Organizes the extracted information as
    id  |   lemma   |   pos |   sense(lemma_from_key)
    xml |   xml     |   xml |   key.txt
    :param xml_path: string. The path to the xml file of the evaluation dataset
    :param wnt: dict. dictionary of WordNet tags
    :return: dictionaries of id, lemma, pos, sense for each sentence
    """

    # returns a generator of tuples (<sentence_number>, <xml tree element indicating the word>)
    words_by_sent = RaganatoReadBy.SENTENCE(xml_path)
    words_by_sent = list(words_by_sent)
    # print(words_by_sent)


    # initialize a defaultdict to group words of same sentence
    # {"0": [element1, ]}
    sentences = defaultdict(list)

    for tup in words_by_sent:
        sentences[tup[0]].append(tup[1])

    # In case a word has no id, i.e. is not sense-tagged, replace its None value to string 'None'
    ids = {k: list(map((lambda x: f'{x.attrib.get("id")}'), lst)) for k, lst in sentences.items()}
    lemmas = {k: list(map((lambda x: x.attrib.get("lemma")), lst)) for k, lst in sentences.items()}
    pos_tags = {k: list(map((lambda x: x.attrib.get("pos")), lst)) for k, lst in sentences.items()}

    # None indices
    none_ids = {k: np.where(np.array(val)=='None')[0].tolist() for k, val in ids.items()}
    valid_ids = ids
    valid_lemmas = lemmas
    valid_pos_tags = pos_tags
    for (k1, val1), (k2, val2), (k3, val3) in zip(valid_ids.items(), valid_lemmas.items(), valid_pos_tags.items()):
        for i in reversed(none_ids[k1]):
            del val1[i]
            del val2[i]
            del val3[i]

    senses = {k: list(map(lambda x: wnt[x], lst)) for k, lst in valid_ids.items()}
    # no need to repeat the sentence, since those datasets are only for testing.


    # ###############################################################################
    # Do some checks before writing to testing files
    # ###############################################################################
    count_ids = {k: len(val) for k, val in ids.items()}
    count_lemmas = {k: len(val) for k, val in lemmas.items()}
    count_pos = {k: len(val) for k, val in pos_tags.items()}
    count_senses = {k: len(val) for k, val in senses.items()}

    N = len(count_ids.keys())
    for i in range(N):
       assert count_ids[i] == count_lemmas[i] == count_pos[i] == count_senses[i], "The number of elements is" \
                                                                                  " inconsistent for key {}".format(i)

    assert len(valid_ids) == len(valid_lemmas) == len(valid_pos_tags) == len(senses), "Detected inconsistency while " \
                                                                                      "reading the testing data len(valid_ids) = {} " \
                                                                                      "len(valid_lemmas) = {} len(valid_pos_tags) = {} len(senses) = {}.".format(
        len(valid_ids), len(valid_lemmas), len(valid_pos_tags), len(senses))

    return valid_ids, valid_lemmas, valid_pos_tags, senses


def extract_full_sent(xml_path):
    """
    Extracts the full sentence from the initial documents.
    :param xml_path: the path to the xml path of a testing dataset
    :return: dictionary of all sentences.
    """
    # returns a generator of tuples (<sentence_number>, <xml tree element indicating the word>)
    words_by_sent = RaganatoReadBy.SENTENCE(xml_path)
    words_by_sent = list(words_by_sent)
    # print(words_by_sent)

    # initialize a defaultdict to group words of same sentence
    # {"0": [element1, ]}
    sentences = defaultdict(list)

    for tup in words_by_sent:
        sentences[tup[0]].append(tup[1])

    full_sentences = {k: " ".join(list(map(lambda x: x.text, lst))) for k, lst in sentences.items()}

    return full_sentences