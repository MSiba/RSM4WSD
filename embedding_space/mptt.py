import datetime
import random
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
import networkx as nx
import matplotlib.pyplot as plt
from itertools import groupby
from collections import deque, defaultdict
import itertools


"""
Initial code from: https://stackoverflow.com/questions/36238796/stack-based-modified-preorder-tree-traversal
"""
wurzel = 'food.n.02'
# wurzel = 'seafood.n.01'

G = nx.read_gpickle(path='food_wordnet.gpickle')
#%%

def get_all_children_of(G, root):
    return [v for u, v in G.edges(G) if u == root]

def get_parent_of(G, child):
    return [u for u, v in G.edges(G) if v == child]

#%%
def mptt_stack(tree, node):
    """
    Stack based implementation
    :param tree:
    :param node:
    :return:
    """
    if node not in tree: return
    preorder = deque()

    stack = []
    for child in reversed(tree[node]):
        stack.append([child, True])

    while stack:
        (node, first) = stack.pop()
        preorder.append(node)
        if first:
            stack.append([node, False])
            if node in tree:
                for child in reversed(tree[node]):
                    stack.append([child, True])

    return preorder


def mptt_recurse(G, node, preorder=None):

    if node not in G.nodes(): return
    if preorder is None: preorder = deque()

    preorder.append(node)

    children = get_all_children_of(G, node)
    for child in children:
        # preorder.append(child)
        mptt_recurse(G, child, preorder)
        # preorder.append(child)

    preorder.append(node)

    return preorder

result = mptt_recurse(G, wurzel)

def map2index(q):
    """
    example output: {'alaska_king_crab.n.01': [189, 190], 'albacore.n.01': [161, 162], ...}
    :param q:
    :return:
    """
    idxlist = [{idx: el} for idx, el in enumerate(q)]

    keyfunc = lambda d: next(iter(d.values()))

    sorted(idxlist, key=keyfunc)

    result = {k: [x for d in g for x in d]
            for k, g in itertools.groupby(sorted(idxlist, key=keyfunc), key=keyfunc)}

    synsets = list(result.keys())
    start = []
    end = []

    for l in result.values():
        start.append(l[0])
        end.append(l[1])

    radius = [np.abs(i - j) for i,j in zip(start, end)]

    cx = [(i+j)/2 for i, j in zip(start, end)]

    output = {'synset': synsets,
              'start': start,
              'end': end,
              'radius': radius,
              'cx': cx}

    return output

r = map2index(result)
df = pd.DataFrame(r)

# radius
# center
# dim





