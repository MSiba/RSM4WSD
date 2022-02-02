import ast
import itertools
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn

import re
import csv
import pickle
import utils


"""__author__ = Siba Mohsen"""

# output_info = '../data/output/pwngc.txt'

# csv because the file will contain a list of values, which is better when processing with torchtext
output_info = 'C:/Users/HP/PycharmProjects/RSM4WSD/data/training_datasets/pwngc4torchtext.csv'
instances = pickle.load(open('C:/Users/HP/PycharmProjects/RSM4WSD/data/ulm/instances.bin', 'rb'))
#%%
df_path = "C:/Users/HP/PycharmProjects/RSM4WSD/data/wordnet_dataframes/SPATIAL_WORDNET.pickle"
tags_df = pd.read_pickle(df_path)
#%%
count = 0
needed = 0

def spatial_tag(df, word, synset):

    if synset == "no-synset":
        return "O"

    params = ["l0", "alpha", "l_i", "beta_i", "radius"]
    tag = None
    syn_df = df.loc[df["synset"] == synset]
    try:
        if syn_df.shape[0] == 1:
            tag = syn_df[params].to_numpy()[0]
            print(" There is only one match")
            print("tag = {}".format(tag))
        else:
            try:
                # try to find best match based on synset tag + word
                best_match = syn_df.loc[syn_df["word"] == word]
                tag = best_match[params].to_numpy()[0]
                print("try to find best match based on synset tag + word")
                print("best match = {}".format(best_match))
                print("tag = {}".format(tag))
            except:
                # if there is no best match, take any value of the subset synset
                random_match = syn_df.loc.sample()
                tag = random_match[params].to_numpy()[0]
                print("there is no best match, take any value of the subset synset")
                print("random match = {}".format(random_match))
                print("tag = {}".format(tag))
    except:
        print("Unable to choose the right spatial tag from SPATIAL_WORDNET.pickle")

    return tag

def clean_offset(pwngc_of):
    """
    Extracts the offset and POS from the token annotation.
    # >>> clean_offset("eng-30-02676054-v")
    "02676054-v"
    ------------------------------------------------------------
    eng stands for language English
    30 stands for the WordNet Version 3.0
    02676054 stands for the offset
    v stands for the POS verb
    -------------------------------------------------------------
    :param pwngc_of: token annotation
    :return: offset-pos
    """
    if pwngc_of == "no-annotation":
        return pwngc_of

    return re.sub('eng-30-', '', pwngc_of)

def synset_name(offset):
    if offset == "no-annotation":
        return "no-synset"
    return wn.of2ss(offset).name() #.lemmas()

"""
Script to create a dataset of PWNGC to train on Word Sense Disambiguation

@INPUT
../data/ulm/instances.bin

In data/output/instances.txt, instances are represented as:
having the necessary---eng-30-01580050-a means or skill or know-how or authority to do something
having the necessary means---eng-30-00172710-n or skill or know-how or authority to do something
having the necessary means or skill or know-how---eng-30-05616786-n or authority to do something
having the necessary means or skill or know-how or authority---eng-30-05196582-n to do something

In data/output/pwngc.txt, instances are prepared for training:
['having', 'the', 'necessary', 'means', 'or', 'skill', 'or', 'know-how', 'or', 'authority', 'to', 'do', 'something']	
[('necessary', 'necessary', 'a', '01580050-a'), ('means', 'means', 'n', '00172710-n'), 
('know-how', 'know-how', 'n', '05616786-n'), ('authority', 'authority', 'n', '05196582-n')]

@Output (old)
sentence_tokens, [(target_token, target_lemma, target_pos, clean_offset(token_annotation))]

@Output
token \t POS \t token_lemma \t clean_offset(token_annotation) \t wn.of2ss(clean_offset(token_annotation)).lemmas() \t dataframe_list 

---------------------------------------------------------------------------------------------
how to get WordNet offsets?
wn.lemma_from_key('feebleminded%5:00:00:retarded:00')
wn.synset_from_pos_and_offset('n', 4543158)
wn.of2ss("02676054-v").lemmas()
"""

with open(output_info, 'w', newline='') as infofile:

    writer = csv.writer(infofile, delimiter="\t")

    for instance_id, instance in instances.items():
        print("instance id = ", instance_id)
        print("instance: ", instance)

        sentence_tokens = []
        sentence_lemmas = []
        sentence_pos = []
        annotations = []

        tokenized_sentence = None
        labels = []


        for token in instance.tokens:
            print("token = ", token.text)
            # print("lemma, type = ", token.lemma, type(token.lemma))

            try:
                print("pos, type = ", token.pos, type(token.pos))
            except AttributeError as ae:
                print(ae, "at token = ", token.text)

            # print("synsets, type = ", token.synsets, type(token.synsets))

            # try:
            sentence_tokens.append(token.text)

            # lemma is of type str
            if token.lemma != '':
                sentence_lemmas.append(token.lemma)
            else:
                sentence_lemmas.append("no-lemma")

            try:
                # pos is of type string
                if token.pos is not None:
                    sentence_pos.append(token.pos)
                    print("POS = ", token.pos)
                else:
                    sentence_pos.append('no-pos')
                    print("No POS = NOTHING")
            except AttributeError as ae:
                sentence_pos.append('no-pos')
                print(ae, "at token = ", token.text)

            # synsets are of type set
            if len(token.synsets) != 0:
                annotations.append(list(token.synsets))
            else:
                annotations.append(list({"no-annotation"}))

            # except AttributeError as e:
            #     print(e, "at token = ", token)

            needed += len(token.synsets)

        print("sentence tokens = ", sentence_tokens)
        print("sentence tokens = ", annotations)
        print("sentence tokens = ", sentence_pos)
        print("sentence tokens = ", sentence_lemmas)

        altern_annotations = list(itertools.product(*annotations))
        # since some tokens have many senses:
        for j in range(len(altern_annotations)):
            for i in range(len(sentence_tokens)):
                clean_off = clean_offset(altern_annotations[j][i])
                syn_name = synset_name(clean_off)
                print("synset name", syn_name)
                tag = spatial_tag(tags_df, word=sentence_lemmas[i], synset=syn_name)

                row = [sentence_tokens[i],
                       sentence_lemmas[i],
                       sentence_pos[i],
                       clean_off,
                       syn_name,
                       tag]

                writer.writerow(row)

            writer.writerow('')

        count += 1

        # for (target_lemma,
        #      target_pos,
        #      token_annotation,
        #      sentence_tokens,
        #      training_example,
        #      target_index) in utils.generate_training_instances_v2(sentence_tokens,
        #                                                            sentence_lemmas,
        #                                                            sentence_pos,
        #                                                            annotations):
        #
        #   # tokenized_sentence = sentence_tokens
            # ------new for spatial params
            # if len(target_lemma) == 0:
            #     target_lemma = "unknown-lemma"
            #
            # if target_pos == '':
            #     target_pos = "no-pos"

            # if len(token_annotation) == 0:
            #     token_annotation = "no-annotation"
            #     clean_off = clean_offset(token_annotation)
            #     syn_name = "no-synset-name"
            #     tag = "O"
            # else:
            #     clean_off = clean_offset(token_annotation)
            #     syn_name = synset_name(clean_off)
            #     print("synset name", syn_name)
            #     tag = spatial_tag(tags_df, word=target_lemma, synset=syn_name)
            #
            # row = [sentence_tokens[target_index],
            #        target_lemma,
            #        target_pos,
            #        clean_off,
            #        syn_name,
            #        tag]
            #
            # writer.writerow(row)
            # ------end new for spatial params

            # --begin old stuff
            # labels.append((sentence_tokens[target_index], target_lemma, target_pos, clean_offset(token_annotation)))

            # print('<START>', target_lemma, '\t', target_pos, '\t', token_annotation, '\t', sentence_tokens, '\t',target_index, '<END>')

            # infofile.write('<START>' + target_lemma + '\t' + target_pos + '\t' + token_annotation + '\t' + str(sentence_tokens) + '\t' + str(target_index) + '<END>' + '\n')
            # infofile.write('<START>' + target_lemma + '\t' + target_pos + '\t' + token_annotation + '\t' + str(sentence_tokens) + '\t' + str(target_index) + '<END>' + '\n')

            # count += 1

        # print(tokenized_sentence, '\t', labels, '\n')

        # infofile.write(str(tokenized_sentence) + '\t' + str(labels) + '\n')
        # writer.writerow('')
        #
        # count += 1


assert needed == count


#<START>  target_lemma  target_pos  token_annotation   sentence_tokens   target_index <END>
"""
<START> relate 	 v 	 eng-30-02676054-v 	 ['relating', 'to', 'or', 'applicable', 'to', 'or', 'concerned', 'with', 'the', 'administration', 'of', 'a', 'city', 'or', 'town', 'or', 'district', 'rather', 'than', 'a', 'larger', 'area'] 	 0 <END>
<START> applicable 	 a 	 eng-30-01975448-a 	 ['relating', 'to', 'or', 'applicable', 'to', 'or', 'concerned', 'with', 'the', 'administration', 'of', 'a', 'city', 'or', 'town', 'or', 'district', 'rather', 'than', 'a', 'larger', 'area'] 	 3 <END>
<START> concern 	 v 	 eng-30-02676054-v 	 ['relating', 'to', 'or', 'applicable', 'to', 'or', 'concerned', 'with', 'the', 'administration', 'of', 'a', 'city', 'or', 'town', 'or', 'district', 'rather', 'than', 'a', 'larger', 'area'] 	 6 <END>
<START> city 	 n 	 eng-30-08524735-n 	 ['relating', 'to', 'or', 'applicable', 'to', 'or', 'concerned', 'with', 'the', 'administration', 'of', 'a', 'city', 'or', 'town', 'or', 'district', 'rather', 'than', 'a', 'larger', 'area'] 	 12 <END>
<START> town 	 n 	 eng-30-08672199-n 	 ['relating', 'to', 'or', 'applicable', 'to', 'or', 'concerned', 'with', 'the', 'administration', 'of', 'a', 'city', 'or', 'town', 'or', 'district', 'rather', 'than', 'a', 'larger', 'area'] 	 14 <END>
<START> district 	 n 	 eng-30-08552138-n 	 ['relating', 'to', 'or', 'applicable', 'to', 'or', 'concerned', 'with', 'the', 'administration', 'of', 'a', 'city', 'or', 'town', 'or', 'district', 'rather', 'than', 'a', 'larger', 'area'] 	 16 <END>
<START> large 	 a 	 eng-30-01382086-a 	 ['relating', 'to', 'or', 'applicable', 'to', 'or', 'concerned', 'with', 'the', 'administration', 'of', 'a', 'city', 'or', 'town', 'or', 'district', 'rather', 'than', 'a', 'larger', 'area'] 	 20 <END>
<START> area 	 n 	 eng-30-08497294-n 	 ['relating', 'to', 'or', 'applicable', 'to', 'or', 'concerned', 'with', 'the', 'administration', 'of', 'a', 'city', 'or', 'town', 'or', 'district', 'rather', 'than', 'a', 'larger', 'area'] 	 21 <END>
"""


# ['having', 'the', 'necessary', 'means', 'or', 'skill', 'or', 'know-how', 'or', 'authority', 'to', 'do', 'something']
# [('necessary', 'necessary', 'a', '01580050-a'), ('means', 'means', 'n', '00172710-n'), ('know-how', 'know-how', 'n', '05616786-n'), ('authority', 'authority', 'n', '05196582-n')]
# from '01580050-a' into Synset.name()?
# how to get WordNet offsets?
# wn.lemma_from_key('feebleminded%5:00:00:retarded:00')
# wn.synset_from_pos_and_offset('n', 4543158)
# wn.of2ss("02676054-v").lemmas()

# with open("C:/Users/HP/PycharmProjects/RSM4WSD/data/output/pwngc.txt", "r") as file:
#     # pwngc = eval(file.readline())
#     # pwngc = file.readline()
#     for line in file:
#         fields = line.split('\t')
#         tokenized_sentence, labels = ast.literal_eval(fields[0]), ast.literal_eval(fields[1])
#         print(tokenized_sentence, labels)
