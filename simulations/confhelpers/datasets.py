import numpy as np


DATASET_MAPPINGS = {
    "path2_spiral": "http://cs.joensuu.fi/sipu/datasets/spiral.txt",
}


def generate_square_manifold(M=200):
    """
    Generates a manifold that is encompassed by a square in 2D
    :param M: The number of input patterns to generate.
    :return: Numpy array with dimensions (M, 2)
    """
    return np.concatenate(
        (
            np.random.uniform(low=1.0, high=3.0, size=(1, M)),
            np.random.uniform(low=1.0, high=3.0, size=(1, M))
        ),
        axis=0
    ).T
