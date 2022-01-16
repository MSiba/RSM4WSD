import networkx as nx
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt


# Experimenting on 1 synset only
wurzel = "freshwater_fish.n.01"
# wurzel = 'seafood.n.01'
# initialize graph G
G = nx.Graph()

# WordNet's POS:
POS = ['n', 'v', 'a', 'r']

vertices = []
edges = []
vertices.append(wurzel)


for word in wn.synset(wurzel).hyponyms():
    vertices.append(word.name())
    # here I can add another for loop for the next dimension
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
    # try:
    #     for hypernym in synset.hypernyms():
    #         edges.append(tuple((hypernym.name(), node)))
    # except AttributeError:
    #     continue
    try:
        for hyponym in synset.hyponyms():
            edges.append(tuple((node, hyponym.name())))
    except AttributeError:
        continue

#%%
print(edges)
print(len(edges))  # 308202 edges

#%%
# To garantee edge order, we need to

G.add_edges_from(edges)


#%%
# store the graph in pickle
nx.write_gpickle(G=G, path='./small_example_wordnet.gpickle')

#%%
# read the graph
G = nx.read_gpickle(path='./small_example_wordnet.gpickle')
#%%
# -----------------------------------------------------------------------
# plot the network
def plot_network(G):

    plt.figure(figsize=(12, 12))

    pos = nx.spring_layout(G, seed=1734289230)

    nx.draw(G, with_labels=True,
            node_color='skyblue',
            edge_cmap=plt.cm.Blues,
            pos = pos)
    nx.draw_networkx_edge_labels(G, pos=pos)
    plt.show()

# ----------------------------------------------------------------
#%%
print([v for u, v in G.edges(G) if u == 'freshwater_fish.n.01'])
