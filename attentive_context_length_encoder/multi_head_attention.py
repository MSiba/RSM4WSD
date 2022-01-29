import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")

"""
The output I will get from the embedding space is as follows: [l0, alpha, l_i, beta_i, radius_i]
this input is saved in a file of the form: [list of words], [list of tags]
                                   Not as:  {"word": the word,
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


https://towardsdatascience.com/augmenting-neural-networks-with-constraints-optimization-ac747408432f
https://en.wikipedia.org/wiki/Conditional_random_field#:~:text=Conditional%20random%20fields%20(CRFs)%20are,can%20take%20context%20into%20account.
label: [POS, l0, alpha, li, beta_i, radius]
if POS == n/v/a/d:
    [POS, ...] is known
    if this word is in wordnet database:
        [POS, l0, alpha,???] is known
    else:
        [POS, ???] is known
else:
    [tree_tagPOS, nothing] because it has only a POS tagging, e.g. The, in and other stop words
    
There are multiple ways how to embed these constraints within the optimization of the attention mechanism.
Which method to choose? --> add a layer? FOL?


Question: is there a way to predict the l_i, beta_i, and radius based on previously detected scores?
https://towardsdatascience.com/what-is-teacher-forcing-3da6217fed1c 
https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/
- multi-output regression is what my labels look like
- I need to teacher force the POS or/and the l0, alpha
- 
"""

"""https://nlp.seas.harvard.edu/2018/04/03/attention.html"""

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None):
    """
    Implements the scaled dot product of Vaswani's Transformer
    :param query: size d_k
    :param key: size d_k
    :param value: size d_v
    :param mask: optional parameter
    :return:
    """

    d_k = query.size(-1)

    # matmul + scale
    query_key = torch.matmul(query, key.transpose(-2,-1)) / torch.sqrt(d_k)

    # mask (Opt.)
    if mask is not None:
        query_key = query_key.masked_fill(mask==0, -1e9)

    # apply softmax
    prob = F.softmax(query_key, dim=-1)

    # matmul prob and value
    result = torch.matmul(prob, value)

    return result, prob


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "h is number of heads and d_model is model size"
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # assume d_v is always equal to d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # linear projections
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
                             for l, x in zip(self.linears, (query, key, value))]

        # attention
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # concatenate
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        # lut: look up table
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             - (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)


    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module --> residual connection"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self,x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self attention and a feed forward network"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


