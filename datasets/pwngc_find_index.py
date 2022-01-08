import pickle
from pprint import pprint as pp

# TODO: search by offset, lemma key
# https://stackoverflow.com/questions/37641584/how-to-get-sense-key-in-wordnet-for-nltk-python

instances = pickle.load(open('../data/ulm/instances.bin', 'rb'))
pp(instances.items())

synset_indices = pickle.load(open('../data/ulm/synset_index.bin', 'rb'))
pp(synset_indices.items())

sense_keys = pickle.load(open('../data/ulm/sensekey_index.bin', 'rb'))
pp(sense_keys.items())


