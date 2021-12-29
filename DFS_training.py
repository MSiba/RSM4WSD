from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
import math
import sys
import wordnet_input
import helper_functions
from pprint import pprint as pp
import networkx as nx
from networkx import pagerank
from nltk.corpus import wordnet as wn
import pickle
import matplotlib.pyplot as plt
from array import array
import matplotlib

"""
In this script, I call the initial functions: adjust2contain and adjust2disconnect and the train one family algorithm
Known issues:
- the algorithms are correct, however I am using recursive calls which is not very pythonic
- especially when calling guess_D_coo from adjust2disconnect --> stackoverflow because no coordinates could be found! 
"""

# Experimenting on 1 synset only
wurzel = "freshwater_fish.n.01"

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

# -----------------------------------------------------------------------
# plot the network
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


#%%
def create_sphere(syn_name,min_value=-15, max_value=15, Ndim=2):
    return {"synset": syn_name,
            "center": np.random.randint(min_value, max_value, size=(1, Ndim))[0],
            "radius": 1.0}

def synset_df(G):
    all_spheres = []
    for vertex in list(G.nodes):
        all_spheres.append(create_sphere(vertex))
    return pd.DataFrame(all_spheres)
#%%
DATAFRAME = synset_df(G)
#%%
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -----------------------------------------------------------------------

def get_all_children_of(G, root):
    return [v for u, v in G.edges(G) if u == root]

def sphere_dist(sphere1, sphere2):
    """
    distance between the centers of two spheres w.r.t. their radii
    if dist < 0 --> they are disconnected and sphere_dist the distance between them
    if dist = 0 --> they are exactly connected
    if dist > 0 --> they are overlapping/same
    :param sphere1:
    :param sphere2:
    :return:
    """
    return (sphere1["radius"] + sphere2["radius"] - np.linalg.norm(sphere1["center"]-sphere2["center"]))

def adjust2contain(G, root_sphere, children):
    """
    to adjust the centers of the spheres s.t. they contain one another
    :param root:
    :param children:
    :return:
    """
    subset_dataframe = DATAFRAME[DATAFRAME["synset"].isin(children)]  # [create_sphere(child) for child in children]
    children_spheres = subset_dataframe.to_dict(orient='records')

    sum_radii = subset_dataframe["radius"].sum()  # TODO: delete /2
    # print("sum_radii = {}".format(sum_radii))
    avg_centers = subset_dataframe["center"].mean()
    # assert avg_centers==np.NAN, "The mean center is NAN"
    avg_centers = np.array([np.round(avg_centers[0],1), np.round(avg_centers[1],1)])
    # print("Average_center", avg_centers, type(avg_centers))

    #enlarge the root sphere such that its radius could contain all spheres
    root_sphere = helper_functions.enlarge(root_sphere, sum_radii)
    root_sphere["center"] = avg_centers
    DATAFRAME.loc[DATAFRAME["synset"] == root_sphere["synset"], ["center", "radius"]] = [[root_sphere["center"]],
                                                                                       root_sphere["radius"]]

    for ind, child_sph in enumerate(children_spheres):
        # print("look here")
        distance_loss = helper_functions.L_P(child_sph, root_sphere)
        print("Containing <{}> in <{}>".format(child_sph["synset"], root_sphere["synset"]))
        # print("distance loss", distance_loss)
        while distance_loss != 0:
            trans_vec, child_sph = helper_functions.guess_P_coo(child_sph, root_sphere)#, VAR=distance_loss)
            children_spheres[ind] = child_sph
            babies_child = get_all_children_of(G, child_sph)
            if len(babies_child) > 0:
                adjust2contain(G, child_sph, babies_child)
            distance_loss = helper_functions.L_P(child_sph, root_sphere)


    for child_sph in children_spheres:
        DATAFRAME.loc[DATAFRAME["synset"] == child_sph["synset"], ["center", "radius"]] = [[child_sph["center"]], child_sph["radius"]]

    return DATAFRAME #root_sphere, children_spheres
# bass_mom = adjust2contain(G, DATAFRAME.loc[6], get_all_children_of(G, "freshwater_bass.n.01"))



def adjust2disconnect(G, root_sphere, children):
    """

    :param G:
    :param children:
    :return: returns all children of one family, such that they are disconnected from each other
    """
    root_center = root_sphere["center"]

    # print("Adjust 2 disconnect")
    # print("children", children)
    nb_children = len(children)
    subset_dataframe = DATAFRAME[DATAFRAME["synset"].isin(children)] #[create_sphere(child) for child in children]
    children_spheres = subset_dataframe.to_dict(orient='records')
    # upper triangular matrix
    # np.triu(matrix, 1)
    U = np.zeros((nb_children, nb_children))
    VAR = 0.5

    # sph_ch = []


    for i, child_i in enumerate(children_spheres):
        for j, child_j in enumerate(children_spheres):
            # print(j, child_j)
            if child_i["synset"] != child_j["synset"] and i < j:
                print("Disconnection <{}> and <{}>".format(child_i["synset"], child_j["synset"]))

                loss = helper_functions.L_D(child_i, child_j)

                while loss != 0:
                    print("1")
                    # if dist_matrix >= 0: # overlaping/same
                    children_child_i = get_all_children_of(G, child_i["synset"])
                    children_child_j = get_all_children_of(G, child_j["synset"])
                    # find distance they need to move from each other

                    if len(children_child_i) == 0:
                        print("2")
                        # length = np.linalg.norm(child_i["center"] - child_j["center"]) - child_j["radius"] - VAR

                        # for index, angle in enumerate(np.linspace(0, 7 * np.pi / 4, 1)):
                        #     child_i = helper_functions.rotate(child_i, root_center, angle) #TODO: need to make dynamic
                        #     # child_i = helper_functions.translate(child_i, VAR)
                        #     # child_i = helper_functions.reduce(child_i, loss)# + VAR)
                        #     loss = helper_functions.L_D(child_i, child_j)
                        #     if loss == 0:
                        #         children_spheres[i] = child_i
                        #     else:
                        #         angle+=10
                        trans, child_i = helper_functions.guess_D_coo(child_i, child_j, mother_sphere=root_sphere)#, VAR=loss)
                        children_spheres[i] = child_i
                        loss = helper_functions.L_D(child_i, child_j)
                    else:
                        print("3")

                        if len(children_child_j) == 0:
                            print("4")

                            # # length = np.linalg.norm(child_i["center"] - child_j["center"]) - child_i["radius"] - VAR
                            # child_j = helper_functions.rotate(child_j, root_center, np.pi)  # TODO: need to make dynamic
                            # # child_j = helper_functions.reduce(child_j, loss + VAR)
                            trans, child_j = helper_functions.guess_D_coo(child_j, child_i, mother_sphere=root_sphere)#, conf=loss)
                            children_spheres[j] = child_j
                            loss = helper_functions.L_D(child_i, child_j)

                        else:
                            print("5")

                            if len(children_child_i) <= len(children_child_j):
                                print("6")
                                try:

                                    # print("guessing child i")
                                    trans_vec, child_i = helper_functions.guess_D_coo(child_i, child_j, mother_sphere=root_sphere)#, VAR=loss)
                                    children_spheres[i] = child_i

                                    # to shift all children
                                    # adjust2disconnect(G, root_sphere, children_child_i)
                                    # adjust2disconnect(G, root_sphere, children_child_j)
                                    # for ch in children_child_i:
                                    #     sph_ch = DATAFRAME[DATAFRAME["synset"]==ch].to_dict(orient="records")[0]
                                    #     sph_ch = helper_functions.translate(sph_ch, trans_vec)
                                    #     DATAFRAME.loc[
                                    #         DATAFRAME["synset"] == sph_ch["synset"], ["center", "radius"]] = [[sph_ch["center"]], sph_ch["radius"]]

                                    loss = helper_functions.L_D(child_i, child_j)
                                except OSError as err:
                                    print("adjust2disconnect is not able to disconnect {} from {}".format(child_i,
                                                                                                          child_j))
                                    print("OSError: {}".format(err))


                                # idea: calculate dist between old and new center as a vector and translate all baby spheres by adding this vector to them
                            else:
                                print("7")

                                try:
                                # print("guessing child j")
                                    trans_vec, child_j = helper_functions.guess_D_coo(child_j, child_i, mother_sphere=root_sphere)#, VAR=loss)
                                    children_spheres[j] = child_j
                                    # shift all children
                                    # adjust2disconnect(G, root_sphere, children_child_j)
                                    # adjust2disconnect(G, root_sphere, children_child_i)
                                    # for ch in children_child_j:
                                    #     sph_ch = create_sphere(ch)
                                    #     sph_ch = helper_functions.translate(sph_ch, trans_vec)
                                    #     DATAFRAME.loc[
                                    #         DATAFRAME["synset"] == sph_ch["synset"], ["center", "radius"]] = [[sph_ch["center"]], sph_ch["radius"]]
                                    loss = helper_functions.L_D(child_i, child_j)
                                except OSError as err:
                                    print("adjust2disconnect is not able to disconnect {} from {}".format(child_j, child_i))
                                    print("OSError: {}".format(err))
                    adjust2disconnect(G, child_i, children_child_i)
                    adjust2disconnect(G, child_j, children_child_j)
                    # U[i][j] = sphere_dist(child_i, child_j)
                    print("8")

            # children_spheres[i] = child_i
            # children_spheres[j] = child_j
    # adjust2disconnect(G, root_sphere,children)
    # print("9")
    for child_sph in children_spheres:
        # print("child_sph", child_sph)
        # print(DATAFRAME)
        # print("values", child_sph.values())
        # print(list(child_sph.values()))
        DATAFRAME.loc[DATAFRAME["synset"] == child_sph["synset"], ["center", "radius"]] = [[child_sph["center"]], child_sph["radius"]]

        # print(DATAFRAME)

                # else:
                #     continue

    return DATAFRAME # TODO: keep it the list of dict?


    #     print(U)
    #
    #
    # # if there is an element in the matrix having a value >= 0, then those spheres must be disconnected:
    # # either by reducing their radii, if they do not already contain other synsets (never do something if they contain other synsets)
    # # or by moving their centers a bit (need to recheck dist to other spheres)
    # # --> translation, rotation
    #
    # # anyways, I need to recheck the new positions
    # rows, cols = np.where(U < 0) # must be <= 0, but I think that I need to re-iterate through U
    # if len(rows) != 0:
    #     print("I need to recheck the disconnectness of spheres after adjustment")
    #
    # for i in range(U.shape[0]):
    #     for j in range(U.shape[1]):
    #         if i<j and U[i][j] == 0:
    #             print("Children ({},{}) are exactly connected to each other. ---> Please re-adjust them".format(i,j))




def create_parent_sphere(sphere, children, EXT=1.0):
    """
    creates the minimum sphere enclosing all children spheres
    https://people.inf.ethz.ch/emo/DoctThesisFiles/fischer05.pdf
    https://doc.cgal.org/latest/Bounding_volumes/classCGAL_1_1Min__sphere__of__spheres__d.html

    :param sphere:
    :param children:
    :param EXT: extension for radius
    :return:
    """
    sum_radii = 0.0
    subset_dataframe = DATAFRAME[DATAFRAME["synset"].isin(children)]  # [create_sphere(child) for child in children]
    children_sph = subset_dataframe.to_dict(orient='records')
    for child_sph in children_sph:
        sum_radii += child_sph["radius"]
    sphere["radius"] = sum_radii + EXT
    DATAFRAME.loc[DATAFRAME["synset"] == sphere["synset"], ["center", "radius"]] = [[sphere["center"]], sphere["radius"]]

    return sphere

def adjust4disconnect(G, root_sphere, children):
    """

    :param G:
    :param children:
    :return: returns all children of one family, such that they are disconnected from each other
    """
    root_center = root_sphere["center"]

    # print("Adjust 2 disconnect")
    # print("children", children)
    nb_children = len(children)
    subset_dataframe = DATAFRAME[DATAFRAME["synset"].isin(children)] #[create_sphere(child) for child in children]
    children_spheres = subset_dataframe.to_dict(orient='records')
    # upper triangular matrix
    # np.triu(matrix, 1)
    U = np.zeros((nb_children, nb_children))
    VAR = 0.5

    # sph_ch = []


    for i, child_i in enumerate(children_spheres):
        for j, child_j in enumerate(children_spheres):
            # print(j, child_j)
            if child_i["synset"] != child_j["synset"] and i < j:
                print("Disconnection <{}> and <{}>".format(child_i["synset"], child_j["synset"]))

                loss = helper_functions.L_D(child_i, child_j)

                while loss != 0:


                    # children_child_i = get_all_children_of(G, child_i["synset"])
                    # children_child_j = get_all_children_of(G, child_j["synset"])
                    # find distance they need to move from each other

                    trans, child_i = helper_functions.guess_D_coo(child_i, child_j, mother_sphere=root_sphere)#, VAR=loss)
                    children_spheres[i] = child_i
                    # adjust4disconnect(G, child_i, children_child_i)
                    # adjust4disconnect(G, child_j, children_child_j)
                    loss = helper_functions.L_D(child_i, child_j)


                # adjust4disconnect(G, child_i, children_child_i)
                # adjust4disconnect(G, child_j, children_child_j)
                # U[i][j] = sphere_dist(child_i, child_j)
                print("8")

    for child_sph in children_spheres:
        DATAFRAME.loc[DATAFRAME["synset"] == child_sph["synset"], ["center", "radius"]] = [[child_sph["center"]], child_sph["radius"]]

    return DATAFRAME  # TODO: keep it the list of dict?



def test_P(G, root_sphere):

    children = get_all_children_of(G, root_sphere["synset"])

    subset_dataframe = DATAFRAME[DATAFRAME["synset"].isin(children)]
    children_spheres = subset_dataframe.to_dict(orient='records')

    loss = np.zeros((len(children)))

    if len(children) > 0:
        for i, child_sphere in enumerate(children_spheres):
            loss[i] = helper_functions.L_P(child_sphere, root_sphere)
            if loss[i] != 0:
                print("<{}> is not part of <{}> with loss = {}".format(child_sphere["synset"], root_sphere["synset"], loss[i]))
            else:
                print("<{}> is part of <{}>".format(child_sphere["synset"], root_sphere["synset"]))
    return loss #zip(children_spheres, loss)



def training_one_family(G, root):
    children = get_all_children_of(G, root)
    root_sphere = DATAFRAME[DATAFRAME["synset"]==root].to_dict(orient="records")[0]

    if len(children) > 0:
        # adjust2contain(G, root_sphere, children)
        for child in children:
            training_one_family(G, child)
            # print("Training One Fam of child {}".format(child))
            # print(training_one_family(G, child))
        if len(children) > 1:
            # adjust2contain(G, root_sphere, children)
            # children_spheres = adjust2disconnect(G, root_sphere, children)
            adjust2disconnect(G, root_sphere, children)
        adjust2contain(G, root_sphere, children)#[0]
        # adjust2disconnect(G, root_sphere, children)
        print("Test PO: ", test_P(G, root_sphere))
        root_sphere = DATAFRAME[DATAFRAME["synset"]==root].to_dict(orient="records")[0]
    else:
        root_sphere = root_sphere
    # DATAFRAME.loc[DATAFRAME["synset"] == root_sphere["synset"], ["center", "radius"]] = [[root_sphere["center"]], root_sphere["radius"]]

    return DATAFRAME # TODO: return children spheres

# small_fam = training_one_family(G, "freshwater_bass.n.01")
result = training_one_family(G, wurzel)
# print(result)

def contain_then_disconnect(G, root_sphere, children):
    adjust2contain(G, root_sphere, children)
    adjust2disconnect(G, root_sphere, children)
    return DATAFRAME
# cd = contain_then_disconnect(G, DATAFRAME.loc[6], get_all_children_of(G, "freshwater_bass.n.01"))
# result = contain_then_disconnect(G, DATAFRAME.loc[0], get_all_children_of(G,wurzel) )
#%%

def draw_circle(sphere):

    word = sphere["synset"]
    center = sphere["center"]
    radius = sphere["radius"]

    figure, axes = plt.subplots()
    axes.set_aspect(1)
    axes.scatter(center[0], center[1], s=10)
    axes.text(center[0], center[1], s=word, color="b")
    plt.title(word)
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plot_circle = plt.Circle(center, radius, edgecolor="b", fill=False)
    axes.add_artist(plot_circle)

    # N, r = 200, .1
    # cms = matplotlib.cm
    # maps = [cms.jet, cms.gray, cms.autumn]

    plt.show()
    return
# draw_circle(result.to_dict(orient="records")[0])

#%%
def visualize(df):
    figure, axes = plt.subplots()
    # axes.set_aspect(1)

    plt.title("The graph structure")
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])

    for i in range(df.shape[0]):
        sphere = df.to_dict(orient="records")[i]
        # draw_circle(sphere)
        word = sphere["synset"]
        center = sphere["center"]
        radius = sphere["radius"]

        axes.scatter(center[0], center[1], s=10)
        axes.text(center[0], center[1], s=word, color="b")

        plot_circle = plt.Circle(center, radius, edgecolor="b", fill=False)
        axes.add_artist(plot_circle)

    plt.show()


visualize(result)
# visualize(bass_mom)
# visualize(cd)






def get_children(synset):
    """
    returns a list of all children nodes of synset
    :param synset: wordnet synset
    :return:
    """
    # node = wn.synset(synset)
    # children = synset.hyponyms()

    # return children
    return

def DFS_train(root):

    # get all hyponyms of root
    children = get_children(root)

    # use depth-first search on each of the synsets for hyponyms also
    # for child in children:
    # train the children w.r.t. their radii and position in space
    # enlarge the radius of root s.t. the root sphere imports all the centers of all hyponyms



    return # must return the new locations


DFS_train("saltwater_fish.n.01")
