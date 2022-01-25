import datetime
import random
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
import networkx as nx
import matplotlib.pyplot as plt
from itertools import groupby
from collections import deque, defaultdict


"""
Current status:
[solved]nested representation works but I need to optimize the hoping between 2 parents in 1 recursion
nx.shortest_path_length(G, wurzel)
try out how to solve the problem with depth

[] the problem happens when the final processed child has children, then in 

start, there are 2 stars and 
end, there is only one [number] between 2 stars
"""

# wurzel = 'food.n.02'
wurzel = 'seafood.n.01'
# wurzel= "freshwater_fish.n.01"
# G = nx.read_gpickle(path='old_seafood_example_wordnet.gpickle')
# #%%
# G.add_nodes_from([wurzel])
# # G.add_edges_from([sorted((i,j)) for i, j in zip(['food.n.02'], ['seafood.n.01'])])
# G.add_edges_from([tuple((wurzel, 'seafood.n.01'))])

G = nx.read_gpickle(path='directed_seafood_wordnet.gpickle')
#%%

def get_all_children_of(G, root):
    return [v for u, v in G.edges(G) if u == root]

def get_parent_of(G, child):
    return [u for u, v in G.edges(G) if v == child]


#%%
visited = []
revisited = []
start = [] #np.zeros(len(G.nodes))
end = []
hierarchy = nx.shortest_path_length(G, wurzel)


def nested_traversal(G, root, step=1, sibling_hop=0):


    visited.append(root)
    start.append(step + sibling_hop)

    step = step + 1 + sibling_hop

    children = get_all_children_of(G, root)

    if len(children) > 0:

        for k, child in enumerate(children):
            nested_traversal(G, root=child, step=step, sibling_hop=k)
            step += 1
        parent = get_parent_of(G, children[0])[0]
        revisited.append(parent)
        revisited.append("*")
        step += len(children)
        end.append(step)
        end.append("*")
        visited.append("*")
        start.append("*")


    else:
        revisited.append(root)
        end.append(step)

# nested_traversal(G, wurzel)
#%%

for i, j, k, l in zip(visited, start, revisited, end):
    print(i,"----------->", j, "   ", k , "~~~~~~~~~~~", l)

# for i, j in zip(revisited, end):
#     print(i, "~~~~~~~~~~~", j)
#%%
# def postprocess_path(start, end):
#     new_start = []
#     new_end = []
#     for i in range(len(start)):
#         if start[i] == "*":
#             diff = end[i-1] - start[i+1] + 1
#             # change start and end until next *
#             for j in range(len(start)):
#                 if j > i:
#                     while start[j] != "*":
#                         start[j] += diff
#                         end[j] += diff
#                     i = j
#%%

split_start = [list(family) for k, family in groupby(start, lambda x: x == "*") if not k]

split_end = [list(family) for k, family in groupby(end, lambda x: x == "*") if not k]

print(split_start)
print(split_end)

#%%
def split_by_star(l):
    return [list(family) for k, family in groupby(l, lambda x: x == "*") if not k]

def flatten(llist):
    return [item for sublist in llist for item in sublist]

#%%
def postprocess_tour(start, end, visited, revisited):
    split_start = split_by_star(start)
    split_end = split_by_star(end)
    split_visited = split_by_star(visited)
    split_revisited = split_by_star(revisited)

    for i in range(len(split_end)):
        if i == 1:
            split_end[i-1].append(split_end[i])
            # delete it
            split_revisited[i-1].append(split_revisited[i])

    d = np.abs(len(split_end) - len(split_start))

    # adjust start and end lists
    for i in range(len(split_end) - 1):
        if i < len(split_start) - 1:
            diff = split_end[i][-1] - split_start[i+1][0] + 1
            split_start[i+1] = [ele + diff for ele in split_start[i+1]]
            split_end[i+1] = [ele + diff for ele in split_end[i+1]]
        else:
            diff = split_end[i][-1] + 1
            split_end[i+1] = [ele + diff for ele in split_end[i+1]]

    # flatten the whole
    dic_start, dict_end = dict(zip(flatten(split_visited), flatten(split_start))), dict(zip(flatten(split_revisited), flatten(split_end)))

    return dic_start, dict_end
#%%
syn_start, syn_end = postprocess_tour(start, end, visited, revisited)


#%%
def relational_dataframe(syn_start, syn_end, hierarchy):

    series = {'start': pd.Series(syn_start), 'end': pd.Series(syn_end)}
    df = pd.DataFrame(series)
    return df

#%%
df = relational_dataframe(syn_start, syn_end, hierarchy)