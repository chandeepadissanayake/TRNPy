import numpy as np


def random(D, N):
    """
    Randomly distributes the N pointers in the embedding space of dimension D.
    :param D: The dimension of the embedding space.
    :param N: Number of pointers to return
    :return: Numpy array of dimensions (D, N).
    """

    return np.random.rand(D, N)
