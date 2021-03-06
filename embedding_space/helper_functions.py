import math
import numpy as np
from nltk.corpus import wordnet as wn
from z3 import *
import random

"""
The current status of this script:
1. I am using mostly L_D, L_P, guess_D_coo and guess_P_coo
2. The functions are all correct, however, I am using recursive calls within them, which is not pythonic
3. This is causing problems sometimes (not for all runs) 
4. The guess_D_coo function or adjust2disconnect does not always yield the correct coordinates, e.g. 
    two sibilings may overlap when their mother sphere is adjusted and never disconnected further
5. The children spheres sometimes do not move due to their static programming 
    --> they do not have enough space to move according to our objective functions in z3 checker
TODO: 
i. look for alternatives to z3 and the recursion 
ii. maybe applying the geometric functions as I wanted initially
iii. after I created the mother sphere for a big tree, I must move it sufficiently in space!!
iv. find an efficient solution for running such an algorithm on all the nodes of the graph
v. In this version of my code, I only considered nodes related to each other using hypernym and hyponyms relations,
I need to consider lonely nodes of WN.
"""


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
    # Warning: np.dot is imprecise, better use it with round!
    dot_prod = np.round(np.dot(unit_1, unit_2), 1)
    try:
        angle = np.arccos(dot_prod)
        return angle * 180.0 / np.pi  # in degrees
    except RuntimeError as e:
        print(e)

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


def enlarge(sphere, length):
    sphere["radius"] = sphere["radius"] + length
    return sphere


def reduce(sphere, length):
    sphere["radius"] = sphere["radius"] - length
    return sphere


def L_D(sphere1, sphere2):
    c1 = sphere1["center"]
    c2 = sphere2["center"]
    r1 = sphere1["radius"]
    r2 = sphere2["radius"]

    dist_loss = r1 + r2 - np.linalg.norm(c1 - c2)
    if dist_loss >= 0:
        # print("The overlaping distance between <{}> and <{}> is: {}.".format(sphere1["synset"],
        #                                                                                           sphere2["synset"],
        #                                                                                           dist_loss))
        return dist_loss
    else:
        # print("<{}> is disconnected from <{}>.".format(sphere1["synset"], sphere2["synset"]))
        return 0

def guess_D_coo(sphere, neighbour_sphere, mother_sphere, VAR=0.1, conf=0.2):
    """
    guess the coordinates of the new center of the sphere, such that the two spheres are disconnected by some distance.
    :param sphere: the moving sphere, with center and radius
    :param c2: center of the second, fix sphere
    :param r2: radius of the second, fix sphere
    :param VAR: variable to make sure that the 2 spheres are seperated by a suitable distance
    :param conf: confidence, added to the center constraints to make sure that the solver guesses coordinates in "range" (not too far)
    :return: the moved sphere, with new coordinates
    """
    c1 = sphere["center"]
    r1 = sphere["radius"]

    c2 = neighbour_sphere["center"]
    r2 = neighbour_sphere["radius"]

    x2 = c2[0]
    y2 = c2[1]

    c0 = mother_sphere["center"]
    r0 = mother_sphere["radius"]
    o1 = c0[0]
    o2 = c0[1]

    loss = L_D(sphere, neighbour_sphere)
    vec = []
    conf = r1
    while loss != 0:
        x1 = Real('x1')
        y1 = Real('y1')

        s = Solver()
        s.add(x1 <= o1 + r0 - conf) #x2 + r2 - conf)
        s.add(x1 >= o1 - r0 + conf) #x2 - r2 + conf)
        s.add(y1 <= o2 + r0 - conf) #y2 + r2 - conf)
        s.add(y1 >= o2 - r0 + conf) # y2 - r2 + conf)
        s.add((x1-x2)**2 + (y1-y2)**2 >= (r1 + r2+ VAR)**2)


        if s.check() == sat:
            m = s.model()
            # print(m)

            # to store the new center coordinates
            coo = np.zeros((1, 2))
            for d in m.decls():

                if d.name() == "x1":
                    print("look here x1")
                    print(type(m[d]))
                    print(m[d])
                    print(is_real(m[d]))
                    print(is_rational_value(m[d]))
                    print(is_algebraic_value(m[d]))

                    if not is_algebraic_value(m[d]) and not is_rational_value(m[d]):
                        coo[0][0] = m[d].as_long()
                    else:
                        if is_rational_value(m[d]):
                            num = m[d].numerator_as_long()
                            denom = m[d].denominator_as_long()
                            irr_nb = float(num) / float(denom)
                            coo[0][0] = round(irr_nb, 1)
                        else:
                            if is_algebraic_value(m[d]):
                                i = m[d].approx(20)
                                num = i.numerator_as_long()
                                denom = i.denominator_as_long()
                                irr_nb = float(num) / float(denom)
                                coo[0][0] = round(irr_nb, 1)
                            else:
                                print("Could not find {} for Disconnection".format(d.name()))


                else:
                    print("look here y1 = {}".format(m[d]))
                    print(is_real(m[d]))
                    print(is_rational_value(m[d]))
                    print(is_algebraic_value(m[d]))

                    if not is_algebraic_value(m[d]) and not is_rational_value(m[d]):
                        coo[0][1] = m[d].as_long()
                    else:
                        if is_rational_value(m[d]):
                            num = m[d].numerator_as_long()
                            denom = m[d].denominator_as_long()
                            irr_nb = float(num) / float(denom)
                            coo[0][1] = round(irr_nb, 1)
                        else:
                            if is_algebraic_value(m[d]):
                                i = m[d].approx(20)
                                num = i.numerator_as_long()
                                denom = i.denominator_as_long()
                                irr_nb = float(num) / float(denom)
                                coo[0][1] = round(irr_nb, 1)
                            else:
                                print("Could not find {} for Disconnection".format(d.name()))

            # print("{} = {}".format(d.name(), m[d]))
            vec = c1 - coo[0]
            sphere["center"] = coo[0]
            loss = L_D(sphere, neighbour_sphere)
            # return vec, sphere
            break
        else:
            # # print("z3 checker for confidence = {} failed ...".format(conf))
            # try:
            #     # conf = random.randrange(-2 * np.abs(conf), 2 * np.abs(conf), 1)
            #     VAR += np.round(random.uniform(0, 1),1)
            #     # conf +=  np.round(random.uniform(0, 1),1)
            #     print("Rechecking for variance = {}.".format(VAR))
            # except:
            #     VAR += 0.1
            #     # conf += 0.1
            # # guess_D_coo(sphere, neighbour_sphere, mother_sphere, conf=conf)
            VAR +=0.1
            if VAR <= 2*r0:
                print("Rechecking for variance = {}.".format(VAR))
                guess_D_coo(sphere, neighbour_sphere, mother_sphere, VAR=VAR)
            else:
                print("There is no possibility for disconnection")
                break

    return vec, sphere



# print(guess_coo(sphere={"synset": "hallo",
#                         "radius": 1.0,
#                         "center": np.array([2,1])},
#                 c2=np.array([2.4,0.7]),
#                 r2=2.0
#                 ))
#
# print(translate(sphere={"synset": "hallo",
#                         "radius": 1.0,
#                         "center": np.array([2,1])},
#                 vec=np.array([2,2.8])))


def L_P(sphere1, sphere2):
    """
    calculates loss of sphere1 being part of sphere2
    :param sphere1:
    :param sphere2:
    :return:
    """
    c1 = sphere1["center"]
    c2 = sphere2["center"]
    r1 = sphere1["radius"]
    r2 = sphere2["radius"]

    dist_loss = r1 + np.linalg.norm(c1-c2) - r2
    if dist_loss > 0:
        # print("The distance loss between <{}> and <{}> is: {}.".format(sphere1["synset"],
        #                                                                sphere2["synset"],
        #                                                                dist_loss))
        return dist_loss
    else:
        # print("<{}> is part of <{}>.".format(sphere1["synset"], sphere2["synset"]))
        return 0

def guess_P_coo(sphere, mother_sphere, VAR=0.1, conf=0.1):
    """
    guess the coordinates of the new center of the child sphere, such that the sphere is contained in the other sphere
    https://stackoverflow.com/questions/12598408/z3-python-getting-python-values-from-model
    :param sphere: the moving sphere, with center and radius
    :param c2: center of the second, fix sphere
    :param r2: radius of the second, fix sphere
    :param VAR: variable to make sure that the 2 spheres are seperated by a suitable distance
    :param conf: confidence, added to the center constraints to make sure that the solver guesses coordinates in "range" (not too far)
    :return: the moved sphere, with new coordinates
    """
    c1 = sphere["center"]
    r1 = sphere["radius"]

    c2 = mother_sphere["center"]
    r2 = mother_sphere["radius"]

    x2 = c2[0]
    y2 = c2[1]

    vec = [0, 0]

    # print("Loss calculated in guess_P_coo")
    loss = L_P(sphere, mother_sphere)
    # print(loss)

    print("guess PART OF ***************************")

    # VAR = r1
    conf = r1

    while loss != 0:

        x1 = Real('x1')
        y1 = Real('y1')

        s = Solver()
        s.add(x1 <= x2 + r2 - conf)
        s.add(x1 >= x2 - r2 + conf)
        s.add(y1 <= y2 + r2 - conf)
        s.add(y1 >= y2 - r2 + conf)
        s.add((x1-x2)**2 + (y1-y2)**2 <= (r2 - r1)**2)

        # print("SAT? {}".format(s.check()))
        if s.check()== sat:
            m = s.model()
            # print(m)

            # to store the new center coordinates
            coo = np.zeros((1,2))
            for d in m.decls():

                if d.name() == "x1":
                    print("look here x1")
                    print(type(m[d]))
                    print(m[d])
                    print(is_real(m[d]))
                    print(is_rational_value(m[d]))
                    print(is_algebraic_value(m[d]))

                    if (not is_algebraic_value(m[d]) and not is_rational_value(m[d])):
                        coo[0][0] = m[d].as_long()
                    else:
                        if is_rational_value(m[d]):
                            num = m[d].numerator_as_long()
                            denom = m[d].denominator_as_long()
                            irr_nb = float(num) / float(denom)
                            coo[0][0] = round(irr_nb, 1)
                        else:
                            if is_algebraic_value(m[d]):
                                i = m[d].approx(20)
                                num = i.numerator_as_long()
                                denom = i.denominator_as_long()
                                irr_nb = float(num) / float(denom)
                                coo[0][0] = round(irr_nb, 1)


                else:
                    print("look here y1 = {}".format(m[d]))
                    print(is_real(m[d]))
                    print(is_rational_value(m[d]))
                    print(is_algebraic_value(m[d]))

                    if not is_algebraic_value(m[d]) and not is_rational_value(m[d]):
                        coo[0][1] = m[d].as_long()
                    else:
                        if is_rational_value(m[d]):
                            num = m[d].numerator_as_long()
                            denom = m[d].denominator_as_long()
                            irr_nb = float(num) / float(denom)
                            coo[0][1] = round(irr_nb, 1)
                        else:
                            if is_algebraic_value(m[d]):
                                i = m[d].approx(20)
                                num = i.numerator_as_long()
                                denom = i.denominator_as_long()
                                irr_nb = float(num) / float(denom)
                                coo[0][1] = round(irr_nb,1)

            # break


                # print("{} = {}".format(d.name(), m[d]))
            vec = c1 - coo[0]
            sphere["center"] = coo[0]
            loss = L_P(sphere, mother_sphere)
            print("coo[0] =", coo[0])
            # print("Non-iterable?", vec, sphere)
            # return vec, sphere
            break
        else:
            print("z3 checker for variance = {} failed ...".format(VAR))
            try:
                # conf = random.randrange(-2 * np.abs(conf), 2 * np.abs(conf), 1)
                VAR += np.round(random.uniform(0, 1),1)
                # print("Rechecking for variance = {}.".format(VAR))
            except:
                VAR += 0.1
            guess_P_coo(sphere, mother_sphere, VAR=VAR)

        # loss = L_P(sphere, mother_sphere)
        # if loss == 0:
        #     break
        # else:
        #     continue

    return vec, sphere


