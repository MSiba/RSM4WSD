import math
import numpy as np
from nltk.corpus import wordnet as wn
from z3 import *
import random


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
    # rotate the vector about (center-point) by alpha
    rotation_matrix = [[np.cos(alpha), -np.sin(alpha)],
                       [np.sin(alpha), np.cos(alpha)]]
    vector = np.array(point - center) # I think that no need for this because center = (0,0)

    sphere["center"] = np.dot(point, rotation_matrix)

    return sphere

def enlarge(sphere, length):
    sphere["radius"] = sphere["radius"] + length
    return sphere

def reduce(sphere, length):
    sphere["radius"] = sphere["radius"] - length
    return sphere

def guess_coo(sphere, c2, r2, VAR=0.5, conf=3.0):
    """
    guess the coordinates of the new center of the sphere, such that the two spheres are disconnected by some distance.
    :param sphere: the moving sphere, with center and radius
    :param c2: center of the second, fix sphere
    :param r2: radius of the second, fix sphere
    :param VAR: variable to make sure that the 2 spheres are seperated by a suitable distance
    :param conf: confidence, added to the center constraints to make sure that the solver guesses coordinates in "range" (not too far)
    :return: the moved sphere, with new coordinates
    """
    x1 = Real('x1')
    y1 = Real('y1')

    c1 = sphere["center"]
    r1 = sphere["radius"]

    x2 = c2[0]
    y2 = c2[1]

    s = Solver()
    s.add(x1 > x2 - conf)
    s.add(x1 < x2 + conf)
    s.add(y1 > y2 - conf)
    s.add(y1 < y2 + conf)
    s.add((x1-x2)**2 + (y1-y2)**2 == (r1 + r2 + VAR)**2)


    if s.check()== sat:
        m = s.model()
        # print(m)

        # to store the new center coordinates
        coo = np.zeros((1,2))
        for d in m.decls():

            if d.name() == "x1":
                # print("look here")
                # print(type(m[d]))
                # print(m[d])
                # print(is_real(m[d]))
                # print(is_rational_value(m[d]))
                # print(is_algebraic_value(m[d]))

                if not is_algebraic_value(m[d]):
                    coo[0][0] = m[d].as_long()
                else:
                    i = m[d].approx(20)
                    num = i.numerator_as_long()
                    denom = i.denominator_as_long()
                    irr_nb = float(num) / float(denom)
                    coo[0][0] = round(irr_nb,1)
            else:
                # print(is_real(m[d]))
                # print(is_rational_value(m[d]))
                # print(is_algebraic_value(m[d]))
                if not is_algebraic_value(m[d]):
                    coo[0][1] = m[d].as_long()
                else:
                    i = m[d].approx(20)
                    num = i.numerator_as_long()
                    denom = i.denominator_as_long()
                    irr_nb = float(num) / float(denom)

                    coo[0][1] = round(irr_nb,1)

            # print("{} = {}".format(d.name(), m[d]))
        vec = c1 - coo[0]
        sphere["center"] = coo[0]
        return vec, sphere
    else:
        print("z3 checker for confidence = {} failed ...".format(conf))
        conf = random.randrange(-2 * conf, 2 * conf, 1)
        print("Rechecking for confidence = {}.".format(conf))
        guess_coo(sphere, c2, r2, conf=conf)




print(guess_coo(sphere={"synset": "hallo",
                        "radius": 1.0,
                        "center": np.array([2,1])},
                c2=np.array([2.4,0.7]),
                r2=2.0
                ))

print(translate(sphere={"synset": "hallo",
                        "radius": 1.0,
                        "center": np.array([2,1])},
                vec=np.array([2,2.8])))


