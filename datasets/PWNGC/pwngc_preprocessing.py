import ast
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn



# ['having', 'the', 'necessary', 'means', 'or', 'skill', 'or', 'know-how', 'or', 'authority', 'to', 'do', 'something']
# [('necessary', 'necessary', 'a', '01580050-a'), ('means', 'means', 'n', '00172710-n'), ('know-how', 'know-how', 'n', '05616786-n'), ('authority', 'authority', 'n', '05196582-n')]
# from '01580050-a' into Synset.name()?
# how to get WordNet offsets?
# wn.lemma_from_key('feebleminded%5:00:00:retarded:00')
# wn.synset_from_pos_and_offset('n', 4543158)
# wn.of2ss("02676054-v").lemmas()

with open("C:/Users/HP/PycharmProjects/RSM4WSD/data/output/pwngc.txt", "r") as file:
    # pwngc = eval(file.readline())
    # pwngc = file.readline()
    for line in file:
        fields = line.split('\t')
        tokenized_sentence, labels = ast.literal_eval(fields[0]), ast.literal_eval(fields[1])
        print(tokenized_sentence, labels)
