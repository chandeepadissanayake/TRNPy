import numpy as np


def random(D, N, dlows=None, dhighs=None):
    """
    Randomly distributes the N pointers in the embedding space of dimension D.
    :param D: The dimension of the embedding space.
    :param N: Number of pointers to return
    :param dlows: Added for compliance with other functions. Not used.
    :param dhighs: Added for compliance with other functions. Not used.
    :return: Numpy array of dimensions (D, N).
    """

    return np.random.rand(D, N)


def uniform(D, N, dlows, dhighs):
    """
    Uniformly distributes the N pointers in the embedding space of dimension D.
    :param D: The dimension of the embedding space
    :param N: Number of pointers to return
    :param dlows: A list of low values to be used in the distribution for every i'th dimension of D dimensions. Should be of length D
    :param dhighs: A list of high values to be used in the distribution for every i'th dimension of D dimensions. Should be of length D.
    :return: Numpy array of dimensions (D, N)
    """

    d = None
    for i in range(D):
        if d is None:
            d = np.random.uniform(low=dlows[i], high=dhighs[i], size=(1, N))
        else:
            d = np.concatenate((d, np.random.uniform(low=dlows[i], high=dhighs[i], size=(1, N))), axis=0)

    return d
