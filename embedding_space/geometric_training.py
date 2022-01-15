import random
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
import networkx as nx
import matplotlib.pyplot as plt

from helper_functions import L_P, L_D
import geometric_functions as gf
from wordnet_graph import plot_network

G = nx.read_gpickle(path='example_wordnet.gpickle')
plot_network(G)


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

def adjust2contain(G, root_sphere, children, keep=False):
    """
    to adjust the centers of the spheres s.t. they contain one another
    :param root:
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
        print("Containing <{}> in <{}>".format(child_sph["synset"], root_sphere["synset"]))

        # calculate the translation vector, by which we need to shift child sphere to mother sphere
        trans_v = np.array(root_sphere["center"] - child_sph["center"])
        print("Translation vector ", trans_v)
        eps = random.uniform(-0.5, 0.5)
        flex_trans_v = trans_v - child_sph["radius"] - eps
        print("Flexible Translation vector: ", flex_trans_v)
        children_spheres[ind] = child_sph

        while distance_loss != 0:
            # trans_vec, child_sph = helper_functions.guess_P_coo(child_sph, root_sphere)#, VAR=distance_loss)
            child_sph = gf.translate(child_sph, flex_trans_v)
            # children_spheres[ind] = child_sph
            # babies_child = get_all_children_of(G, child_sph)

            children_spheres[ind] = child_sph

            if len(babies_child) > 0:
                adjust2contain(G, child_sph, babies_child)
            distance_loss = L_P(child_sph, root_sphere)


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

    VAR = 0.5
    ETA = 0.1

    # sph_ch = []


    for i, child_i in enumerate(children_spheres):
        for j, child_j in enumerate(children_spheres):
            # print(j, child_j)
            if child_i["synset"] != child_j["synset"]:# and i < j:
                print("Disconnection <{}> and <{}>".format(child_i["synset"], child_j["synset"]))

                # radius1 = child_i["radius"]
                # radius2 = child_j["radius"]

                loss = L_D(child_i, child_j)

                while loss != 0:
                    print("1")
                    children_child_i = get_all_children_of(G, child_i["synset"])
                    children_child_j = get_all_children_of(G, child_j["synset"])
                    # find distance they need to move from each other

                    if len(children_child_i) == 0:
                        print("2")
                        child_i = gf.rotate_arclength(child_i, child_j, root_sphere, loss, ETA=ETA)
                        children_spheres[i] = child_i
                        loss = L_D(child_i, child_j)
                        print("loss = ", loss)
                        if loss != 0:
                            ETA += random.uniform(0,1)
                            transvec = np.array([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)])
                            child_i = gf.translate(child_i, transvec)
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
                            transvec = np.array([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)])
                            child_j = gf.translate(child_j, transvec)
                            continue
                        else:
                            break


                    if len(children_child_i) <= len(children_child_j):
                        print("6")
                        # try:

                        child_i = gf.rotate_arclength(child_i, child_j, root_sphere, loss, ETA=ETA)

                        children_spheres[i] = child_i
                        loss = L_D(child_i, child_j)
                        print("loss = ", loss)

                        if loss != 0:
                            ETA += random.uniform(0,1)
                            transvec = np.array([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)])
                            child_i = gf.translate(child_i, transvec)
                            continue
                        else:
                            # adjust2contain(G, child_i, children_child_i, keep=True)
                            break

                    if len(children_child_i) > len(children_child_j):
                        print("7.1")
                        child_j = gf.rotate_arclength(child_j, child_i, root_sphere, loss, ETA=ETA)
                        children_spheres[j] = child_j
                        loss = L_D(child_i, child_j)
                        print("loss = ", loss)

                        if loss != 0:
                            ETA += random.uniform(0,1)
                            transvec = np.array([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)])
                            child_j = gf.translate(child_j, transvec)

                            continue
                        else:
                            # adjust2contain(G, child_j, children_child_j, keep=True)
                            break


                    # adjust2disconnect(G, child_i, children_child_i)
                    # adjust2disconnect(G, child_j, children_child_j)

                    adjust2contain(G, child_i, children_child_i, keep=True)
                    adjust2contain(G, child_j, children_child_j, keep=True)

                    print('8')
                loss = L_D(child_i, child_j)
                print("loss = ", loss)

    # check for after processing them all if some are still connected to each other
    children, test = test_D(G, root_sphere)
    if np.all((test == 0)):
        pass
    else:
        print("WARNING: there are connected sibling spheres.")
        # adjust2disconnect(G, root_sphere, children)

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
            if loss[i] > 0:
                print("<{}> is not part of <{}> with loss = {}".format(child_sphere["synset"], root_sphere["synset"], loss[i]))
            else:
                print("<{}> is part of <{}>".format(child_sphere["synset"], root_sphere["synset"]))
    return children, loss


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
                    if loss[i][j] != 0:
                        print("<{}> is overlaping <{}> with loss = {}".format(child_i["synset"], child_j["synset"], loss[i][j]))
                    else:
                        print("<{}> is distant from <{}>".format(child_i["synset"], child_j["synset"]))


    return children, loss

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
            adjust2contain(G, root_sphere, children, keep=True)
            adjust2shift(G, root_sphere, children)

        print("Test PO: ", test_P(G, root_sphere))

        print("Test DC: ", test_D(G, root_sphere))
    visualize(DATAFRAME)
    print("DATAFRAME Test PO: ", test_P(G, root_sphere))

    print("DATAFRAME Test DC: ", test_D(G, root_sphere))
    return DATAFRAME


#TODO:
def training_all_families(G):
    return

def testing_all_families(G):
    return

# TODO: storage?
# TODO: centers? barymetric center? mean?

wurzel = "freshwater_fish.n.01"

# small_fam = training_one_family(G, "freshwater_bass.n.01")
result = training_one_family(G, wurzel)
# print(result)


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
