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

# TODO: add later
semcor_spacial_tags = None

semcor_df = pd.DataFrame(data={'sentence':semcor_sentences, 'sense_tags': semcor_tags})

#%%
sent = semcor_tags[0]
print(sent)
words = []
POS = []
senses = []
for l in sent:
    if isinstance(l, nltk.tree.Tree):
        words.append(l.leaves())
        POS.append(l.pos())
        senses.append(l.label())
    else:
        None
# def tagging2list(sentence):
#     words_list =
# semcor_df.to_pickle("./data/semcor_data.pickle")
# save into pickle to save computational cost
#%%
semcor_df = pd.read_pickle("./data/semcor_dataset.pickle")
# import semcor
# import PWNGC
# wrap them in Dataset iterator using torchtext