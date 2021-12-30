"""
The Input I will get from the embedding space is as follows: [l0, alpha, l_i, beta_i, radius_i]
this input is saved in a file of the form: {"word": the word,
                                            "synset": WN synset,
                                            "POS":,
                                            "offset":,
                                            "definition":,
                                            "examples":,
                                            "l0": word_point,
                                            "alpha": word_angle,
                                            "l_i": sense_length,
                                            "beta_i": sense_orientation,
                                            "radius_i": sense_relations}
Training data: WordNet PWNGC (+ SemCor)
For loading PWNGC into python classes: https://github.com/cltl/pwgc

I will begin with the simplest architecture:
- an input x which is encoded into Embedding(x) using GloVe/word2vec or FLAIR/ELMO or random initialization then to train it?
- single head vs. multihead attention (need to know what are exactly the weight matrices and how they are trained exactly)
- FFNN vs. Fully Connected Linear Layer
- align and normalize
- softmax and finally output
"""
