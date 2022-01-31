import string
import pandas as pd
import nltk

nltk.download('semcor')
from nltk.corpus import semcor

#%%
# get sentences from Semcor as tokenized lists
# 37176 sentences
semcor_sentences = semcor.sents()

# get the tagged senses for each sentence
# tag is set to "both" to get POS + semantic tagging
semcor_tags = semcor.tagged_sents(tag='both')
semcor_senses = semcor.tagged_sents(tag='sem')
semcor_pos = semcor.tagged_sents(tag='pos')

# TODO: add later
semcor_spacial_tags = None
#%%
# import preprocessing_utils as pu
#
# sents, tags = pu.semcor_preprocessing(semcor_sentences, semcor_tags)
#
# #%%
# print(len(sents[0]), len(tags[0]))
# print(sents[0], "\n", tags[0])
#%%
semcor_df = pd.DataFrame(data={'sentence': semcor_sentences, 'sense_tags': semcor_tags})

#%%
# TODO: after I know, how is the attention mechanism working exactly, I need to adjust the Semcor dataset in a way that it fits to the attention input.
#TODO: decide what to include in the dataset exactly, Pay attention if the position of word in the sentence plays a role.
# TODO: we can have the same word in a sentence with 2 different meanings.

# wn.lemma('dog.n.01.dog').synset()
# >> Synset('dog.n.01')
# To store the Lemma name only,
# [str(lemma.name()) for lemma in wn.synset('dog.n.01').lemmas()]
# >> ['dog', 'domestic_dog', 'Canis_familiaris']

sent = semcor_tags[0]
print(sent)
words = []
POS = []
senses = []
for l in sent:
    if isinstance(l, nltk.tree.Tree):
        print(l.leaves())
        words.append(l.leaves())
        POS.append(l.pos())
        if not isinstance(l.label(), str):
            senses.append(l.label())
        else:
            senses.append(None)
    else:
        words.append(None)
        POS.append(None)
        # senses.append(None)
#%%
def distorte_tags(semcor_tag, NER=False):
    words = []
    POS = []
    senses = []
    if NER:
        for l in semcor_tag:
            if isinstance(l, nltk.tree.Tree):
                words.append(l.leaves())
                POS.append(l.pos())
                if not isinstance(l.label(), str):
                    senses.append(l.label())
                else:
                    senses.append(None)
            else:
                words.append(None)
                POS.append(None)
                senses.append(None)
    else:
        for l in semcor_tag:
            if isinstance(l, nltk.tree.Tree):
                if len(l.leaves()) > 1:
                    for i, entity in enumerate(l.leaves()):
                        words.append([entity])
                        POS.append([l.pos()[i]])
                        if not isinstance(l.label(), str):
                            senses.append(l.label())
                        else:
                            senses.append(None)
                else:
                    words.append(l.leaves())
                    POS.append(l.pos())
                    if not isinstance(l.label(), str):
                        senses.append(l.label())
                    else:
                        senses.append(None)
            else:
                words.append(None)
                POS.append(None)
                senses.append(None)
    assert len(words) == len(POS) == len(senses), "Processing different length sequences ({},{},{})".format(len(words), len(POS), len(senses))

    return {"word_list": words,
            "POS_list": POS,
            "lemma_list": senses}

#%%
print(distorte_tags(sent))
print(distorte_tags(sent, NER=False))
# semcor_df = pd.DataFrame(data={'sentence':semcor_sentences, 'sense_tags': semcor_tags})
#%%
additional_col = [distorte_tags(semcor_tags[i], NER=False) for i in range(semcor_df.shape[0])]
# semcor_df.append()
#%%
len(additional_col)
postprocess_df = pd.DataFrame(additional_col)
SEMCOR_DATAFRAME = pd.concat([semcor_df, postprocess_df], axis=1)
#%%
# do some statistics on semcor dataframe, e.g. missing fields?
print(SEMCOR_DATAFRAME.isnull().values.any())

#%%
# to pickle
# SEMCOR_DATAFRAME.to_pickle("./data/semcor_dataframe.pickle")
# pickle does not work correctly and feather because the values are all lists
SEMCOR_DATAFRAME.to_json("./data/semcor_dataframe.json")

#%%
#TODO: store in ulm format!
# https://wiki.fysik.dtu.dk/ase/ase/io/ulm.html
# change lemma to offset ID

# check if I can read semcor from pickle
# test = pd.read_pickle("./data/semcor_dataframe.pickle")

# semcor_df.to_pickle("./data/semcor_data.pickle")
# save into pickle to save computational cost
#%%
# semcor_df = pd.read_pickle("./data/semcor_dataset.pickle")
# import semcor
# import PWNGC
# wrap them in Dataset iterator using torchtext