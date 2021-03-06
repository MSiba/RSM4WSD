import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import training


def visualize(word, color="k"):
    """
    plots [l0, alpha, li, beta, radius] in 2D
    :param word_senses: information about [l0, alpha, li, beta, radius]
    :return:
    """
    params, synsets = training.initialize(word)


    center = params["center"]
    l0 = params["l0"]
    alpha = params["alpha"]
    syn_coo = params["synsets_coo"]
    li = params["li"]
    betas = params["betas"]
    radii = params["radii"]

    ax = plt.gca()
    ax.set_aspect("equal")

    ax.scatter(center[0], center[1], color=color, s=50)

    for coo in syn_coo:
        print(coo)
        ax.scatter(coo[0], coo[1], color=color, s=50)

    origin = np.array([[0, 0, 0], [0, 0, 0]])  # origin point

    plt.quiver(*origin, center[0], center[1], color="k", angles='xy', scale_units='xy', scale=1)

    for coo in syn_coo:
        plt.quiver(*center, coo[0], coo[1], color="m", angles='xy', scale_units='xy', scale=1)


    plt.show()

# visualize("bass")

def plot_network(G):

    plt.figure(figsize=(12, 12))

    pos = nx.spring_layout(G, seed=1734289230)

    nx.draw(G, with_labels=True,
            node_color='skyblue',
            edge_cmap=plt.cm.Blues,
            pos = pos)
    nx.draw_networkx_edge_labels(G, pos=pos)
    plt.show()