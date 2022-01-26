import networkx as nx
from nltk.corpus import wordnet as wn
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt
#%%
"""
For verbs: they do have a hierarchy, but not like that of nouns with entity.n.01 as mother
https://stackoverflow.com/questions/36060492/nltk-wordnet-verb-hierarchy

Statistics on WordNet 3.0: http://manpages.ubuntu.com/manpages/trusty/man7/wnstats.7WN.html
https://stackoverflow.com/questions/28876407/how-to-find-the-lexical-category-of-a-word-in-wordnet-using-nltkpython/28908041
"""
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

"""
There are 82115 nouns, not only 74374
check out other relations types / include them simply in space
wn.all_synsets(pos='n')

to get all the missing synsets, use the relation instance_hyponomy()
No need for meronomy because it will yield 2 different parents in our case  
"""


def extract_hyponyms(root_synset):
    vertices = [root_synset]
    edges = []

    # for word in wn.synset(root_synset).hyponyms():
    #     vertices.append(word.name())

    # new_vertices = []

    for node in vertices:
        synset = wn.synset(node)

        try:
            print("hyponym of {}".format(node))
            for hyponym in synset.hyponyms():
                # add this hyponym to the nodes of wordnet
                vertices.append(hyponym.name())
                # relate hyponym with edge
                edges.append(tuple((node, hyponym.name())))
            # extract_hyponyms(node)
            # -----------------------------------
            # INSTANCE HYPONYMS
            print("instance hyponym of {}".format(node))

            instance_hypo = synset.instance_hyponyms()
            if len(instance_hypo) != 0:
                for instance in instance_hypo:
                    vertices.append(instance.name())
                    edges.append(tuple((node, instance.name())))

            # # -----------------------------------
            # # PART MERONYMS
            # print("part meronym?".format(node))
            # part_mer = synset.part_meronyms()
            # if len(part_mer) != 0:
            #     print("part meronym of {}".format(node))
            #     for meronym in part_mer:
            #         vertices.append(meronym.name())
            #         edges.append(tuple((node, meronym.name())))
            # # --------------------------------------------
            # member_mer = synset.member_meronyms()
            # if len(member_mer) != 0:
            #     print("member meronym of {}".format(node))
            #     for mem_meronym in member_mer:
            #         vertices.append(mem_meronym.name())
            #         edges.append(tuple((node, mem_meronym.name())))
            # # ------------------------------------------------
            # print("substance meronym?".format(node))
            # substance_mer = synset.substance_meronyms()
            # if len(substance_mer) != 0:
            #     print("substance meronym of {}".format(node))
            #     for sub_meronym in substance_mer:
            #         vertices.append(sub_meronym.name())
            #         edges.append(tuple((node, sub_meronym.name())))

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
# nx.write_gpickle(G=WN, path='./wordnet_nouns.gpickle')

#%%
# G = nx.read_gpickle(path="./wordnet.gpickle")
# #%%
# # VERBS
# # 13.767 verb
# verbs = [verb for verb in wn.all_synsets(pos='v')]
# #%%
# # verbs don't have one common hypernym
# # there are verbs having over 1000 hyponym, others only 1, namely, themselves
# # to embed them, I will need the mppt algorithm to run over each tree out of 559 trees
# #
# root_hypernyms_verbs = Counter(chain(*[ss.root_hypernyms() for ss in wn.all_synsets(pos='v')]))
# verb_roots = [k.name() for k in root_hypernyms_verbs.keys()]
# # verb_graphs = [extract_hyponyms(s) for s in verb_roots]
#
# def extract_meronyms(root_synset):
#     """
#     indicates the meronymy relations between words.
#     a meronym denoting a part and a holonym denoting a whole.
#     difference to hyper/hypo: is-a relation indicates the type/kind.
#     example: hyper/hypo: pine_tree is kind of tree
#     meronomy: leaves(mer) are part_of tree(holonym)
#     :param root_synset:
#     :return:
#     """
#     vertices = [root_synset]
#     edges = []
#
#     # for word in wn.synset(root_synset).hyponyms():
#     #     vertices.append(word.name())
#
#     # new_vertices = []
#
#     for node in vertices:
#         synset = wn.synset(node)
#
#         try:
#             for hyponym in synset.hyponyms():
#                 # add this hyponym to the nodes of wordnet
#                 vertices.append(hyponym.name())
#                 # relate hyponym with edge
#                 edges.append(tuple((node, hyponym.name())))
#             # extract_hyponyms(node)
#         except AttributeError:
#             continue
#
#     nodes = vertices #+ new_vertices
#     G = nx.DiGraph()
#     G.add_nodes_from(nodes)
#     G.add_edges_from(edges)
#
#     return G