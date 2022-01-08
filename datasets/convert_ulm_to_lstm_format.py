import pickle
import utils

"""Source: https://github.com/cltl/pwgc"""


output_path = '../data/instances.txt'

instances = pickle.load(open('../data/ulm/instances.bin', 'rb'))


count = 0
needed = 0

with open(output_path, 'w') as outfile:
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

            sentence_tokens.append(token.text)
            sentence_lemmas.append(token.lemma)
            sentence_pos.append(token.pos)
            annotations.append(list(token.synsets))

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
            count += 1


assert needed == count
