import networkx as nx
from networkx import pagerank
from nltk.corpus import wordnet as wn
import pickle

# initialize graph G
G = nx.Graph()

# WordNet's POS:
POS = ['n', 'v', 'a', 'r']

vertices = []
edges = []

# list(wn.all_synsets(pos) for pos in POS)
for pos in POS:
    for word in list(wn.all_synsets(pos)):
        # vertices.append(list(wn.all_synsets(pos)))
        vertices.append(word.name())
#%%
print(len(vertices))
print(vertices)
print(wn.synsets(vertices[0]))
#%%
# add the vertices to the graph
G.add_nodes_from(vertices)
#%%
# edges are the hypernyms, hyponyms, member_holonyms, root_hypernyms, meronym
# hypernym is a word whose meaning includes a group of other words
# hyponym is a word whose meaning is included in the meaning of another one
# meronym is a part-of relation of its holonym, e.g. finger(meronym) is part-of hand(holonym)

# I think that we are not really interested in storing the edge labels, we just need a relation
for node in G.nodes():
    synset = wn.synset(node)
    try:
        for hypernym in synset.hypernyms():
            edges.append((hypernym.name(), node))
    except AttributeError:
        continue
    try:
        for hyponym in synset.hyponyms():
            edges.append((node, hyponym.name()))
    except AttributeError:
        continue
    try:
        for holo in synset.member_holonyms():
            edges.append((node, holo.name()))
    except:
        continue

    try:
        for root in synset.root_hypernyms():
            edges.append((root.name(), node))
    except AttributeError:
        continue

# print(edges)
# print(len(edges))  # 308202 edges
G.add_edges_from(edges)
