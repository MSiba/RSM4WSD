import networkx as nx
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt


# Experimenting on 1 synset only
wurzel = "entity.n.01"
# wurzel = "freshwater_fish.n.01"
# wurzel = 'seafood.n.01'
# wurzel = "food.n.02"
# initialize graph G
# G = nx.Graph()

# WordNet's POS:
POS = ['n', 'v', 'a', 'r']

# vertices = []
# edges = []
# vertices.append(wurzel)


def extract_hyponyms(root_synset):
    vertices = [root_synset]
    edges = []

    # for word in wn.synset(root_synset).hyponyms():
    #     vertices.append(word.name())

    # new_vertices = []

    for node in vertices:
        synset = wn.synset(node)

        try:
            for hyponym in synset.hyponyms():
                # add this hyponym to the nodes of wordnet
                vertices.append(hyponym.name())
                # relate hyponym with edge
                edges.append(tuple((node, hyponym.name())))
            # extract_hyponyms(node)
        except AttributeError:
            continue

    nodes = vertices #+ new_vertices
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G
#%%
WN = extract_hyponyms(wurzel)
#%%
# store graph
# store the graph in pickle
nx.write_gpickle(G=WN, path='./240122_freshwater_wordnet.gpickle')
