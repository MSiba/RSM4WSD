import datetime
import random
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
import networkx as nx
import matplotlib.pyplot as plt

from helper_functions import L_P, L_D
import geometric_functions as gf
from wordnet_graph import plot_network

# "__author__ = Siba Mohsen"

#TODO: use pickle5 for python 3.8+ or arrow to store pickles more efficiently
# https://stackoverflow.com/questions/63329657/python-3-7-error-unsupported-pickle-protocol-5
# https://arrow.apache.org/docs/python/ipc.html#reading-from-stream-and-file-format-for-pandas

begin_time = datetime.datetime.now()

wurzel = 'seafood.n.01'
G = nx.read_gpickle(path='example_wordnet.gpickle')
plot_network(G)

# wurzel = "freshwater_fish.n.01"
# G = nx.read_gpickle(path='small_example_wordnet.gpickle')
# plot_network(G)


"""Current status:
this code works very well for our example,
I just need to check how deep can it go with root the mother of the "freshwater_fish": 'seafood.n.01'
Behaviour: I need to adjust the calls of children, since it is only optimizing one family --> more recursion in adjust2contain and adjust2disconnect
         ** When creating the graph, I need to consider all the nodes till the leaves!
- add tests!
- test for whole graph
- clean 
"""
# ---------------------------------------------------------------------

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

#%%
def adjust2contain(G, root_sphere, children, keep=False):
    """
    to adjust the centers of the spheres s.t. they contain one another
    :param root:+
    :param children:
    :return:
    """
    subset_dataframe = DATAFRAME[DATAFRAME["synset"].isin(children)]  # [create_sphere(child) for child in children]
    children_spheres = subset_dataframe.to_dict(orient='records')

    sum_radii = subset_dataframe["radius"].sum()
    # print("sum_radii = {}".format(sum_radii))
    avg_centers = subset_dataframe["center"].mean()
    # assert avg_centers==np.NAN, "The mean center is NAN"
    avg_centers = np.array([np.round(avg_centers[0],1), np.round(avg_centers[1],1)])
    # print("Average_center", avg_centers, type(avg_centers))

    #enlarge the root sphere such that its radius could contain all spheres
    root_sphere = gf.enlarge(root_sphere, sum_radii)
    root_sphere["center"] = avg_centers
    DATAFRAME.loc[DATAFRAME["synset"] == root_sphere["synset"], ["center", "radius"]] = [[root_sphere["center"]],
                                                                                       root_sphere["radius"]]

    print("Root center", root_sphere["center"])
    for ind, child_sph in enumerate(children_spheres):
        print("Initial child center: ", child_sph["center"])

        # if child sphere has no children, reduce its radius
        babies_child = get_all_children_of(G, child_sph)
        # VAR = random.uniform(0.3, 0.3)
        if not keep:
            if len(babies_child) == 0:
                child_sph = gf.reduce(child_sph, 0.5)
            children_spheres[ind] = child_sph

        # calculate distance loss
        distance_loss = L_P(child_sph, root_sphere)
        # print("Containing <{}> in <{}>".format(child_sph["synset"], root_sphere["synset"]))

        # # calculate the translation vector, by which we need to shift child sphere to mother sphere
        # trans_v = np.array(root_sphere["center"] - child_sph["center"])
        # # print("Translation vector ", trans_v)
        # eps = random.uniform(-0.5, 0.5)
        # flex_trans_v = trans_v - child_sph["radius"] - eps
        # # print("Flexible Translation vector: ", flex_trans_v)
        # # children_spheres[ind] = child_sph

        while distance_loss != 0:
            # calculate the translation vector, by which we need to shift child sphere to mother sphere
            trans_v = np.array(root_sphere["center"] - child_sph["center"])
            # print("Translation vector ", trans_v)
            eps = random.uniform(0, child_sph["radius"])
            flex_trans_v = trans_v - child_sph["radius"] - eps
            # print("Flexible Translation vector: ", flex_trans_v)
            # children_spheres[ind] = child_sph
            # ------------------------------------------------
            # trans_vec, child_sph = helper_functions.guess_P_coo(child_sph, root_sphere)#, VAR=distance_loss)
            child_sph = gf.translate(child_sph, flex_trans_v)
            # children_spheres[ind] = child_sph
            babies_child = get_all_children_of(G, child_sph)
            if len(babies_child) > 0:
                # babies_child is only the synset name, how to
                adjust2map(G, child_sph, babies_child, vector=flex_trans_v)
                # adjust2contain(G, child_sph, babies_child)

            children_spheres[ind] = child_sph

            distance_loss = L_P(child_sph, root_sphere)

        # if len(babies_child) > 0:
        #     # babies_child is only the synset name, how to
        #     adjust2map(G, child_sph, babies_child, vector=flex_trans_v)
        #     # adjust2contain(G, child_sph, babies_child)
            # distance_loss = L_P(child_sph, root_sphere)


    for child_sph in children_spheres:
        DATAFRAME.loc[DATAFRAME["synset"] == child_sph["synset"], ["center", "radius"]] = [[child_sph["center"]], child_sph["radius"]]

    children, P_loss = test_P(G, root_sphere)
    if np.all((P_loss == 0)):
        return DATAFRAME
    else:
        print("WARNING: A child is not part of its mother sphere.")
        adjust2contain(G, root_sphere, children, keep=True)
#%%

def shift_whole_family(G, root, initial_center, current_center, children):

    # def shift_whole_family(G, initial_mother, current_mother, children):
    # TODO: I need to check why it is not working well
    # compare with the simplest version!
    # maybe add it to shift2adjust also?

    # initial_center = initial_mother["center"]
    # current_center = current_mother["center"]

    subset_dataframe = DATAFRAME[DATAFRAME["synset"].isin(children)]
    children_spheres = subset_dataframe.to_dict(orient='records')

    # eps = random.uniform(0, 0.2)

    for k, baby in enumerate(children_spheres):
        initial_baby = baby["center"]
        tmp_vec = np.array(current_center - initial_center) #+ eps
        baby = gf.translate(baby, tmp_vec)
        children_spheres[k] = baby
        children_baby = get_all_children_of(G, baby)
        # if len(children_baby) > 0:
        #     shift_whole_family(G=G,
        #                        initial_center=initial_baby,
        #                        current_center=children_spheres[k]["center"],
        #                        children=children_baby)
        # else:
        #     pass



    for child_sph_i in children_spheres:
        DATAFRAME.loc[DATAFRAME["synset"] == child_sph_i["synset"], ["center", "radius"]] = [
            [child_sph_i["center"]], child_sph_i["radius"]]

    # check if after chifting the whole family, all the children follow their mothers
    babies, P_loss = test_P(G, root)

    if np.all((P_loss == 0)):
        pass
    else:
        for j, baby, ploss in enumerate(zip(babies, P_loss)):
            if ploss != 0:
                vec = np.array(current_center - baby["center"])
                ploss_vec = vec * ploss / np.linalg.norm(vec)
                adjust2map(G, root, [baby], vector=ploss_vec)

    return DATAFRAME

#%%

def adjust2disconnect(G, root_sphere, children, STEP_SIZE=0.5):
    """
    :param G:
    :param children:
    :return: returns all children of one family, such that they are disconnected from each other
    """
    root_center = root_sphere["center"]
    nb_children = len(children)

    subset_dataframe = DATAFRAME[DATAFRAME["synset"].isin(children)]
    children_spheres = subset_dataframe.to_dict(orient='records')

    VAR = 0.5
    ETA = 0.1

    for i, child_i in enumerate(children_spheres):
        for j, child_j in enumerate(children_spheres):

            if child_i["synset"] != child_j["synset"]: # and i < j:

                print("disconnecting <{}> and <{}>".format(child_i["synset"], child_j["synset"]))
                initial_center_i = child_i["center"]
                initial_center_j = child_j["center"]

                loss = L_D(child_i, child_j)

                while loss != 0:
                    print("1")
                    # get the children of each child --> Type: String
                    children_child_i = get_all_children_of(G, child_i["synset"])
                    children_child_j = get_all_children_of(G, child_j["synset"])
                    #
                    # subset_dataframe_i = DATAFRAME[DATAFRAME["synset"].isin(children_child_i)]
                    # children_child_i_spheres = subset_dataframe_i.to_dict(orient='records')
                    #
                    # subset_dataframe_j = DATAFRAME[DATAFRAME["synset"].isin(children_child_j)]
                    # children_child_j_spheres = subset_dataframe_j.to_dict(orient='records')

                    if len(children_child_i) == 0:
                        print("2")
                        child_i = gf.rotate_arclength(child_i, child_j, root_sphere, loss, ETA=ETA)
                        children_spheres[i] = child_i
                        loss = L_D(child_i, child_j)
                        print("loss = ", loss)
                        if loss != 0:
                            ETA += random.uniform(0,1)
                            transvec = np.array([random.uniform(-STEP_SIZE, STEP_SIZE),
                                                 random.uniform(-STEP_SIZE, STEP_SIZE)])
                            child_i = gf.translate(child_i, transvec)
                            # test_part_of = L_P(child_i, root_sphere)
                            # if test_part_of != 0:
                            #     transvec -= test_part_of # TODO:  needs more optimization to update the radius?
                            #     child_i = gf.translate(child_i, transvec)
                            continue
                        else:
                            break

                    if len(children_child_j) == 0:
                        print("4")
                        child_j = gf.rotate_arclength(child_j, child_i, root_sphere, loss, ETA=ETA)
                        # transvector = child_i["center"] - child_j["center"]
                        # child_j = gf.translate(child_j, transvector)
                        children_spheres[j] = child_j
                        loss = L_D(child_i, child_j)
                        print("loss = ", loss)
                        if loss != 0:
                            ETA += random.uniform(0, 1)
                            transvec = np.array([random.uniform(-STEP_SIZE, STEP_SIZE),
                                                 random.uniform(-STEP_SIZE, STEP_SIZE)])
                            child_j = gf.translate(child_j, transvec)
                            # test_part_of = L_P(child_j, root_sphere)
                            # if test_part_of != 0:
                            #     transvec -= test_part_of
                            #     child_j = gf.translate(child_j, transvec)
                            continue
                        else:
                            break


                    if len(children_child_i) <= len(children_child_j):
                        print("6")
                        # try:
                        # old_pos_i = child_i["center"]
                        child_i = gf.rotate_arclength(child_i, child_j, root_sphere, loss, ETA=ETA)
                        # new_pos_i = child_i["center"]
                        children_spheres[i] = child_i
                        # print("Using Map for child i")
                        # pos_vec_i = np.array(new_pos_i - old_pos_i)
                        # adjust2map(G, child_i, children_child_i, pos_vec_i)
                        # let the child do exactly what its parent is doing
                        loss = L_D(child_i, child_j)
                        print("loss = ", loss)

                        if loss != 0:
                            ETA += random.uniform(0,1)
                            transvec = np.array([random.uniform(-STEP_SIZE, STEP_SIZE),
                                                 random.uniform(-STEP_SIZE, STEP_SIZE)])
                            child_i = gf.translate(child_i, transvec)
                            # test_part_of = L_P(child_i, root_sphere)
                            # if test_part_of != 0:
                            #     transvec -= test_part_of
                            #     child_i = gf.translate(child_i, transvec)
                            continue
                        else:
                            shift_whole_family(G=G,
                                               root=child_i,
                                               initial_center=initial_center_i,
                                               current_center=children_spheres[i]["center"],
                                               children=children_child_i)
                            # babies, P_loss = test_P(G, child_i)
                            # if np.all((P_loss == 0)):
                            #     print("all babies of child i are correctly adjusted during disconnection")
                            # else:
                            #     for baby, ploss in zip(babies, P_loss):
                            #         if ploss != 0:
                            #             vec = np.array(child_i["center"] - baby["center"]) -2*baby["radius"]-loss
                            #             adjust2map(G, child_i, [baby], vector=vec)
                            #     print("Some babies of child i of the root sphere are damaged after disconnection")
                            #     print("root sphere: ", root_sphere["synset"])
                            #     print("child_i: ", child_i["synset"])
                            #     print("Babies and their losses: \n", babies, "\n", P_loss)
                            break

                    if len(children_child_i) > len(children_child_j):
                        print("7.1")
                        # old_pos_j = child_j["center"]
                        child_j = gf.rotate_arclength(child_j, child_i, root_sphere, loss, ETA=ETA)
                        # new_pos_j = child_j["center"]
                        children_spheres[j] = child_j

                        # print("Using Map for child j")
                        # pos_vec_j = np.array(new_pos_j - old_pos_j)
                        # adjust2map(G, child_j, children_child_j, vector=pos_vec_j)
                        loss = L_D(child_i, child_j)
                        print("loss = ", loss)

                        if loss != 0:
                            ETA += random.uniform(0,1)
                            transvec = np.array([random.uniform(-STEP_SIZE, STEP_SIZE),
                                                 random.uniform(-STEP_SIZE, STEP_SIZE)])
                            child_j = gf.translate(child_j, transvec)
                            # test_part_of = L_P(child_j, root_sphere)
                            # if test_part_of != 0:
                            #     transvec -= test_part_of
                            #     child_j = gf.translate(child_j, transvec)
                            continue
                        else:
                            shift_whole_family(G=G,
                                               root=child_j,
                                               initial_center=initial_center_j,
                                               current_center=children_spheres[j]["center"],
                                               children=children_child_j)
                            # babies, P_loss = test_P(G, child_j)
                            # if np.all((P_loss == 0)):
                            #     print("all babies of child j are correctly adjusted during disconnection")
                            # else:
                            #     for baby, ploss in zip(babies, P_loss):
                            #         if ploss != 0:
                            #             #TODO: baby sphere, not only name
                            #             vec = np.array(child_j["center"] - baby["center"]) -2 * baby["radius"]-loss
                            #             adjust2map(G, child_j, [baby], vector=vec)
                            #     print("Some babies of child j of the root sphere are damaged after disconnection")
                            #     print("root sphere: ", root_sphere["synset"])
                            #     print("child_j: ", child_j["synset"])
                            #     print("Babies and their losses: \n", babies, "\n", P_loss)

                            break

                    # as long as I am doing break in the else statement, I am just skipping this part and jumping to the loss directly
                    # (ending loop)

                    # adjust2disconnect(G, child_i, children_child_i)
                    # adjust2disconnect(G, child_j, children_child_j)

                    # adjust2contain(G, child_i, children_child_i, keep=True)
                    # adjust2contain(G, child_j, children_child_j, keep=True)

                    # adjust2map(G, child_i, children_child_i)
                    # adjust2map(G, child_j, children_child_j)

                    print('8')
                loss = L_D(child_i, child_j)
                print("loss = ", loss)


    for child_sph in children_spheres:
        DATAFRAME.loc[DATAFRAME["synset"] == child_sph["synset"], ["center", "radius"]] = [[child_sph["center"]], child_sph["radius"]]


    # check for after processing them all if some are still connected to each other
    children, test = test_D(G, root_sphere)
    if np.all((test == 0)):
        return DATAFRAME
    else:
        print("WARNING: there are connected sibling spheres.")
        adjust2disconnect(G, root_sphere, children)

    # return DATAFRAME

def adjust2map(G, mother_sphere, children, vector):
    """
    Let children follow their mother sphere
    :param G:
    :param mother_sphere:
    :param children:
    :param vector:
    :return:
    """

    subset_dataframe = DATAFRAME[DATAFRAME["synset"].isin(children)]  # [create_sphere(child) for child in children]
    children_spheres = subset_dataframe.to_dict(orient='records')


    for i, child in enumerate(children_spheres):
        # c1 = child["center"]
        # r1 = child["radius"]
        child = gf.translate(child, vector)
        # child["center"] = c1 - c0 + np.array([random.uniform(-limit, limit),
        #                                       random.uniform(-limit, limit)])
        children_spheres[i] = child


    for child_sph in children_spheres:
        DATAFRAME.loc[DATAFRAME["synset"] == child_sph["synset"], ["center", "radius"]] = [[child_sph["center"]], child_sph["radius"]]

    return DATAFRAME


def adjust2shift(G, root_sphere, children, ratio=3):
    """
    In case a root sphere has only 1 child, this child is always contained exactly on the center.
    adjust2shift shifts the child a bit
    :param G:
    :param root_sphere:
    :param child:
    :return:
    """
    c0 = root_sphere["center"]
    r0 = root_sphere["radius"]

    # vector = c0 + (r0/2)
    vector = np.array([r0/ratio, r0/ratio])
    print("Vector", vector, type(vector))
    subset_dataframe = DATAFRAME[DATAFRAME["synset"].isin(children)]
    print("Subset df", subset_dataframe)
    children_spheres = subset_dataframe.to_dict(orient='records')
    print("children spheres", children_spheres)

    for index, child in enumerate(children_spheres):
        child = gf.translate(child, vector)
        children_spheres[index] = child

    for child_sph in children_spheres:
        DATAFRAME.loc[DATAFRAME["synset"] == child_sph["synset"], ["center", "radius"]] = [[child_sph["center"]], child_sph["radius"]]


    return DATAFRAME

def create_parent_sphere(sphere, children, EXT=1.0):
    """
    creates the minimum sphere enclosing all children spheres
    https://stackoverflow.com/questions/9063453/how-to-compute-the-smallest-bounding-sphere-enclosing-other-bounding-spheres
    https://people.inf.ethz.ch/emo/DoctThesisFiles/fischer05.pdf
    https://doc.cgal.org/latest/Bounding_volumes/classCGAL_1_1Min__sphere__of__spheres__d.html
    https://stackoverflow.com/questions/3102547/how-can-i-find-the-minimal-circle-include-some-given-points


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


def test_P(G, root_sphere):

    children = get_all_children_of(G, root_sphere["synset"])

    subset_dataframe = DATAFRAME[DATAFRAME["synset"].isin(children)]
    children_spheres = subset_dataframe.to_dict(orient='records')

    loss = np.zeros((len(children)))

    if len(children) > 0:
        for i, child_sphere in enumerate(children_spheres):
            loss[i] = L_P(child_sphere, root_sphere)
            # if loss[i] > 0:
            #     print("<{}> is not part of <{}> with loss = {}".format(child_sphere["synset"], root_sphere["synset"], loss[i]))
            # else:
            #     print("<{}> is part of <{}>".format(child_sphere["synset"], root_sphere["synset"]))
    return children_spheres, loss


def test_D(G, root_sphere):

    children = get_all_children_of(G, root_sphere["synset"])

    subset_dataframe = DATAFRAME[DATAFRAME["synset"].isin(children)]
    children_spheres = subset_dataframe.to_dict(orient='records')

    loss = np.zeros((len(children), len(children)))


    if len(children) > 1:
        for i, child_i in enumerate(children_spheres):
            for j, child_j in enumerate(children_spheres):
                if child_i["synset"] != child_j["synset"] and i < j:

                    loss[i][j] = L_D(child_i, child_j)
                    # if loss[i][j] != 0:
                    #     print("<{}> is overlaping <{}> with loss = {}".format(child_i["synset"], child_j["synset"], loss[i][j]))
                    # else:
                    #     print("<{}> is distant from <{}>".format(child_i["synset"], child_j["synset"]))


    return children, loss
#%%
def visualize(df, name="<name>"):
    figure, axes = plt.subplots()
    # axes.set_aspect(1)

    plt.title("processing {}".format(name))
    plt.xlim([-30, 30])
    plt.ylim([-40, 40])

    for i in range(df.shape[0]):
        sphere = df.to_dict(orient="records")[i]
        # draw_circle(sphere)
        word = sphere["synset"]
        center = sphere["center"]
        radius = sphere["radius"]

        try:
            axes.scatter(center[0], center[1], s=10)
            axes.text(center[0], center[1], s=word, color="b")
        except IndexError:
            axes.scatter(center[0][0], center[0][1], s=10)
            axes.text(center[0][0], center[0][1], s=word, color="b")


        plot_circle = plt.Circle(center, radius, edgecolor="b", fill=False)
        axes.add_artist(plot_circle)

    plt.show()
#%%

def training_one_family(G, root):
    children = get_all_children_of(G, root)
    root_sphere = DATAFRAME[DATAFRAME["synset"]==root].to_dict(orient="records")[0]

    if len(children) > 0:
        for child in children:
            training_one_family(G, child)

        adjust2contain(G, root_sphere, children)

        if len(children) > 1:
            adjust2disconnect(G, root_sphere, children)

        if len(children) == 1:
            # adjust2contain(G, root_sphere, children, keep=True)
            adjust2shift(G, root_sphere, children)

    # visualize(DATAFRAME, name="<{}>".format(root_sphere["synset"]))

    return DATAFRAME

#%%
def test_one_family(G, df, root):
    """
    Tests two things, (1) its disconnectness from sibling spheres, and (2) all its children are part of it
    :param G:
    :param df:
    :param root:
    :return:
    """

    children = get_all_children_of(G, root)

    for child in children:
        test_one_family(G, df, child)

    root_sphere = df[df["synset"] == root].to_dict(orient="records")[0]

    # to test containment of all children in root sphere
    babies, P_loss = test_P(G, root_sphere)
    # to test disconnectness of all children in root sphere
    siblings, D_loss = test_D(G, root_sphere)

    print("Testing node <{}>".format(root))

    if np.all((P_loss == 0)):
        print("Hurra! All its babies are contained in it!")
    else:
        # print("Not all its babies are contained in it.")
        print("Babies not following their mother by: ")
        print(babies)
        print(P_loss)

    if np.all((D_loss) == 0):
        print("Hurra! All its babies are disconneted from each other!")
    else:
        # print("Some of its babies are connected to each other.")
        print("Babies connected to each other")
        print(siblings)
        print(D_loss)


# test_one_family(G, DATAFRAME, wurzel)
#%%
#TODO:
def training_all_families(G):
    return

def testing_all_families(G):
    return

# TODO: storage?
# TODO: centers? barymetric center? mean?


# small_fam = training_one_family(G, "freshwater_bass.n.01")
result = training_one_family(G, wurzel)
# print(result)


#%%
visualize(result)
# visualize(bass_mom)
# visualize(cd)
#%%
# store DATAFRAME
# DATAFRAME.to_pickle("./seafood.pickle")
#%%
# blabla = pd.read_pickle("./seafood.pickle")


print(datetime.datetime.now() - begin_time)