import numpy as np
import pandas as pd
import wordnet_input
import helper_functions
from pprint import pprint as pp


# To set min and max integer values, math.inf did not work because float, sys.maxsize did not work also
def initialize(word, Ndim=2, min_value=-10, max_value=10):
    """
    initializes the word (center) as well as its senses (vectors) in the embedding space
    :param word:
    :param Ndim: the dimension of coordinates, default is 2D
    :return:
    """
    synsets = wordnet_input.get_senses(word)
    senses_nb = len(synsets["synsets"])
    origin = np.zeros((1, Ndim))
    print(origin)

    # create random center coordinates
    center_int = np.random.randint(min_value, max_value, size=(1, Ndim))
    center_word = center_int + np.random.random((1,Ndim)) # for 2D vector
    print(center_word, type(center_word), center_word.shape)
    # insert coordinates to the center
    synsets["word"]["center"] = center_word

    # create random points for senses
    # to create numbers that are greater than 1 and less than 0, add random integers to the floats
    # No need for them to be different now, because their length will be adjusted
    senses_int = np.random.randint(min_value, max_value, size=(senses_nb, Ndim))
    senses_floats = np.random.random((senses_nb, Ndim))
    senses_coo = senses_int + senses_floats
    # insert coordinates of each synset to the dict
    for i in range(senses_nb):
        synsets["synsets"][i]["coo"] = senses_coo[i]

    print(type(senses_coo), senses_coo.shape, senses_coo)
    print(synsets)

    # calculate vector l0 and the distance between center and origin
    l0_vec = center_word - origin
    print("l0_vec = ", l0_vec)
    l0 = np.linalg.norm(l0_vec)
    # find the angle alpha between x-axis and l0
    x_axis = np.zeros((1, Ndim))
    x_axis[0][0] = 1
    print("x-axis", x_axis[0])
    alpha = helper_functions.find_angle(x_axis[0], l0_vec[0])
    print("alpha = ", alpha)

    # calculate length between center each sense
    li_vecs = center_word - senses_coo
    print(li_vecs)
    l = [np.linalg.norm(li_vec) for li_vec in li_vecs]
    print("l i = ", l)

    # calculate the beta angles with respect to the li_vectors
    betas = [helper_functions.find_angle(l0_vec, li_vecs[i]) for i in range(senses_nb)]
    print(betas)

    # create uniform radiis for all word senses
    r = np.full(shape=senses_nb, fill_value=1.)

    # add the parameters to the center and senses
    # length between origin and center word
    synsets["word"]["l0"] = l0
    # the angle alpha
    synsets["word"]["alpha"] = alpha
    # the lengths of all senses
    for i in range(senses_nb):
        synsets["synsets"][i]["l"] = l[i]
        synsets["synsets"][i]["beta"] = betas[i][0]
        synsets["synsets"][i]["radius"] = r[i]
        # I changed this to do outer join later on
        synsets["synsets"][i]["word"] = synsets["word"]["stem_word"]

    pp(synsets)
    train_params = {"center": center_word[0], # Ndim array
                    "l0": l0, # length float
                    "alpha": alpha,
                    "synsets_coo": senses_coo, # list of Ndim arrays
                    "li": l, # list of lengths
                    "betas": betas, # list of angles
                    "radii": r # list of radii
                    }

    pp(train_params)


    synsets_df = pd.DataFrame(synsets["synsets"])
    print(synsets_df)



    return train_params, synsets
initialize("clothes")

