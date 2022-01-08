import pickle
import utils

"""Source: https://github.com/cltl/pwgc"""


output_path = '../data/output/instances.txt'
output_info = '../data/output/pwngc.txt'

instances = pickle.load(open('../data/ulm/instances.bin', 'rb'))


count = 0
needed = 0

with open(output_path, 'w') as outfile:
    with open(output_info, 'w') as infofile:
        for instance_id, instance in instances.items():
            print("instance id = ", instance_id)
            print("instance: ", instance)

            sentence_tokens = []
            sentence_lemmas = []
            sentence_pos = []
            annotations = []


            for token in instance.tokens:
                print("token = ", token)
                print(type(token))

                try:
                    sentence_tokens.append(token.text)
                    sentence_lemmas.append(token.lemma)
                    sentence_pos.append(token.pos)
                    annotations.append(list(token.synsets))
                except AttributeError as e:
                    print(e, "at token = ", token)

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

                outfile.write(training_example + '\n')
                print('<START>', target_lemma, '\t', target_pos, '\t', token_annotation, '\t', sentence_tokens, '\t',target_index, '<END>')

                # infofile.write('<START>' + target_lemma + '\t' + target_pos + '\t' + token_annotation + '\t' + sentence_tokens + '\t' + target_index + '<END>')
                count += 1


assert needed == count

# TODO: create my own data set as (sentences: [tokens], labels:[senses])
# TODO: how to get WordNet offsets?
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