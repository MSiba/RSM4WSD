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

# TODO: add later
semcor_spacial_tags = None
#%%
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
        if not isinstance(l.label(), str):
            senses.append(l.label())
        else:
            senses.append(None)
    else:
        words.append(None)
        POS.append(None)
        # senses.append(None)

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
# to pickle
SEMCOR_DATAFRAME.to_pickle("./data/semcor_dataframe.pickle")


# semcor_df.to_pickle("./data/semcor_data.pickle")
# save into pickle to save computational cost
#%%
# semcor_df = pd.read_pickle("./data/semcor_dataset.pickle")
# import semcor
# import PWNGC
# wrap them in Dataset iterator using torchtext