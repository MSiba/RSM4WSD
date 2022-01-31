import re
import csv
import pickle
import utils


"""__author__ = Siba Mohsen"""


output_info = '../data/output/pwngc.txt'

instances = pickle.load(open('../data/ulm/instances.bin', 'rb'))


count = 0
needed = 0



# Siba modified this
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

    return re.sub('eng-30-', '', pwngc_of)


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

@Output
sentence_tokens, [(target_token, target_lemma, target_pos, clean_offset(token_annotation))]

---------------------------------------------------------------------------------------------
how to get WordNet offsets?
wn.lemma_from_key('feebleminded%5:00:00:retarded:00')
wn.synset_from_pos_and_offset('n', 4543158)
wn.of2ss("02676054-v").lemmas()
"""

with open(output_info, 'w') as infofile:
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
            print("token = ", token)

            # try:
            sentence_tokens.append(token.text)
            sentence_lemmas.append(token.lemma)
            try:
                sentence_pos.append(token.pos)
                print("POS = ", token.pos)
            except:
                sentence_pos.append('')
                print("No POS = NOTHING")
            annotations.append(list(token.synsets))
            # except AttributeError as e:
            #     print(e, "at token = ", token)

            needed += len(token.synsets)

        for (target_lemma,
             target_pos,
             token_annotation,
             sentence_tokens,
             training_example,
             target_index) in utils.generate_training_instances_v2(sentence_tokens,
                                                                   sentence_lemmas,
                                                                   sentence_pos,
                                                                   annotations):

            tokenized_sentence = sentence_tokens

            labels.append((sentence_tokens[target_index], target_lemma, target_pos, clean_offset(token_annotation)))

            print('<START>', target_lemma, '\t', target_pos, '\t', token_annotation, '\t', sentence_tokens, '\t',target_index, '<END>')

            # infofile.write('<START>' + target_lemma + '\t' + target_pos + '\t' + token_annotation + '\t' + str(sentence_tokens) + '\t' + str(target_index) + '<END>' + '\n')
            # infofile.write('<START>' + target_lemma + '\t' + target_pos + '\t' + token_annotation + '\t' + str(sentence_tokens) + '\t' + str(target_index) + '<END>' + '\n')

            # count += 1

        print(tokenized_sentence, '\t', labels, '\n')

        infofile.write(str(tokenized_sentence) + '\t' + str(labels) + '\n')

        count += 1


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




