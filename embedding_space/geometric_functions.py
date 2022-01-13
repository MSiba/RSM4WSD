import numpy as np
from nltk.corpus import wordnet as wn

def RELATION(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    relational_matrix = np.zeros((len(synsets1), len(synsets2)))
    for i in synsets1:
        for j in synsets2:
            hyper_1 = set.intersection(set())


def find_angle(vec1, vec2):
    unit_1 = vec1/np.linalg.norm(vec1)
    unit_2 = vec2/np.linalg.norm(vec2)
    angle = np.arccos(np.dot(unit_1, unit_2))
    return angle * 180.0 / np.pi # in degrees

# I am still not sure if I want to translate a point, a sphere or a whole subgraph/tree
def translate(sphere, vec): #acts as prolongement
    sphere["center"] = sphere["center"] + vec
    return sphere

def rotate(sphere, center, alpha): # I think I'll mostly use this as the origin (0,0) but for now keep it as it is

    point = sphere["center"]
    # print("Point: {}, Center: {}".format(point, center))
    # rotate the vector about (center-point) by alpha
    rotation_matrix = [[np.cos(alpha), -np.sin(alpha)],
                       [np.sin(alpha), np.cos(alpha)]]
    vector = np.array(point - center)
    sphere["center"] = np.squeeze((rotation_matrix @ (point.T-center.T) + center.T).T)#np.dot(vector, rotation_matrix)

    return sphere

def rotate_arclength(sphere, root_sphere, arclength, ETA=0.1):
    """
    Rotates a sphere overlaping with another sphere with respect to the center point.
    The arclength represents the overlaping distance (calculated by L_D).
    :param sphere:
    :param root_sphere:
    :param arclength:
    :param ETA: to extend the arclength a bit and ensure that the two spheres are not overlaping.
    :return:
    """
    c0 = root_sphere["center"]

    # radius of the circle passing through the sphere center point
    radius = np.linalg.norm(sphere["center"] - c0)
    alpha = (arclength+ETA) / radius

    return rotate(sphere, c0, np.pi - alpha)


def enlarge(sphere, length):
    sphere["radius"] = sphere["radius"] + length
    return sphere


def reduce(sphere, length):
    sphere["radius"] = sphere["radius"] - length
    return sphere