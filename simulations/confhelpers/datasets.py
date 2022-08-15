import numpy as np


DATASET_MAPPINGS = {
    "path2_spiral": "http://cs.joensuu.fi/sipu/datasets/spiral.txt",
}


def generate_square_manifold(M=200, xlow=1.0, ylow=1.0, xhigh=3.0, yhigh=3.0):
    """
    Generates a manifold that is encompassed by a square in 2D
    :param M: The number of input patterns to generate
    :param xlow: The low value to be used w.r.t. x-axis in the uniformly distributed input patterns
    :param ylow: The low value to be used w.r.t. y-axis in the uniformly distributed input patterns
    :param xhigh: The high value to be used w.r.t. x-axis in the uniformly distributed input patterns
    :param yhigh: The high value to be used w.r.t. y-axis in the uniformly distributed input patterns
    :return: Numpy array with dimensions (M, 2)
    """
    return np.concatenate(
        (
            np.random.uniform(low=xlow, high=xhigh, size=(1, M)),
            np.random.uniform(low=ylow, high=yhigh, size=(1, M))
        ),
        axis=0
    ).T
