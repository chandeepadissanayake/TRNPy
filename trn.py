import numpy as np
import random


def adapt(W, V, t_max, T_i, T_f, T_sch, eps_i, eps_f, eps_sch, lmd_i, lmd_f, lmd_sch, verbose=True, output_per_steps=10):
    """
    Build the Topology Preserving Network, homogeneously distributing pointers in W over the given manifold M.
    Utilizes neural gas algorithm and competitive Hebb rule.
    Results in the connection strength matrix.
    :param W: Pointers Matrix - A Numpy array (Matrix) of dimensions (D, N) where D is the dimension of the input pattern space (embedding space), N is the number of pointers
    :param V: Input Patterns Matrix - A Numpy array (Matrix) of dimensions (D, ) where D is the dimension of the input pattern space. Any number of input pattern/columns can be defined and used. Each input pattern must be an element of the manifold M
    :param t_max: Maximum number of iterations that the algorithm is expected to take
    :param T_i: Initial value for lifetime (age to consider a connection has expired)
    :param T_f: Final value for lifetime (age to consider a connection has expired)
    :param T_sch: Scheduler for updating T (utilized in updating connections, based on their age)
                  Should provide a function which accepts 4 parameters:
                        1. T_i: Initial value for T in neural gas algorithm's update rule.
                        2. T_f: Final value for T in neural gas algorithm's update rule.
                        3. t: The iteration step at which the algorithm is currently in.
                        4. t_max: Maximum number of iterations that the algorithm is expected to take.
    :param eps_i: Initial value for epsilon in neural gas algorithm's update rule
    :param eps_f: Final value for epsilon in neural gas algorithm's update rule
    :param eps_sch: Scheduler for updating epsilon at (utilized in update rule of the neural gas algorithm) at each iteration step t.
                    Should provide a function which accepts 4 parameters:
                        1. eps_i: Initial value for epsilon in neural gas algorithm's update rule.
                        2. eps_f: Final value for epsilon in neural gas algorithm's update rule.
                        3. t: The iteration step at which the algorithm is currently in.
                        4. t_max: Maximum number of iterations that the algorithm is expected to take.
    :param lmd_i: Initial value for lambda in neural gas algorithm's update rule
    :param lmd_f: Final value for lambda in neural gas algorithm's update rule
    :param lmd_sch: Scheduler for updating lambda (utilized in update rule of the neural gas algorithm) at each iteration step t.
                    Should provide a function which accepts 4 parameters:
                        1. lmd_i: Initial value for lambda in neural gas algorithm's update rule.
                        2. lmd_f: Final value for lambda in neural gas algorithm's update rule.
                        3. t: The iteration step at which the algorithm is currently in.
                        4. t_max: Maximum number of iterations that the algorithm is expected to take.
    :return: A 2-tuple of Numpy arrays.
             1. Updated Pointers W, with the same dimensions (D, N)
             2. Numpy array (Matrix) of dimensions (N, N) where each element at indices i, j where i != j, provides the connection strength between neural units fixed by a pointer, each.
    """

    N = W.shape[1]
    """ Setting all connection strengths to 0 """
    C = np.zeros((N, N))
    """ Age matrix for all connections """
    A = np.zeros(C.shape)

    for t in range(t_max):
        """ Selecting an input pattern v with uniform probability """
        v_idx = random.randint(0, V.shape[1] - 1)
        v = V[:, v_idx]

        """ Norms of 'distance vectors' """
        vw = (W.T - v).T
        vw_norms = np.linalg.norm(vw, axis=0)

        """ k values for each w """
        k = np.zeros(vw_norms.shape)
        for i, norm_i in enumerate(vw_norms):
            k[i] = np.count_nonzero(vw_norms < norm_i)

        """ Updating epsilon, lambda with the schedulers """
        eps = eps_sch(eps_i, eps_f, t, t_max)
        lmd = lmd_sch(lmd_i, lmd_f, t, t_max)

        """ Updating the pointers by neural gas algorithm's update rule """
        W += eps * (np.exp(-k / lmd) * (v - W.T).T)

        """ Obtaining first and second closes units i0 and i1 """
        i_f2 = np.argpartition(vw_norms, 2)
        i0 = i_f2[0]
        i1 = i_f2[1]

        """ Creating a connection between neural units at i0 and i1 and updating ages """
        C[i0, i1] = 1
        C[i1, i0] = 1  # Just for the sake of maintaining symmetric nature of the matrix.

        A[i0, i1] = 0
        A[i1, i0] = 0  # Just for the sake of maintaining symmetric nature of the matrix.

        """ Increasing the age of all connections of neural unit at i0 """
        A[i0, :] = C[i0, :] * (A[i0, :] + 1)
        A[:, i0] = C[:, i0] * (A[:, i0] + 1)  # Just for the sake of maintaining symmetric nature of the matrix.

        """ Updating lifetime parameter by the scheduler """
        T = T_sch(T_i, T_f, t, t_max)

        """ Removing old connections where T > t_max """
        C[i0, A[i0, :] > T] = 0
        C[A[:, i0] > T, i0] = 0  # Just for the sake of maintaining symmetric nature of the matrix.

        """ Output """
        if verbose and (t + 1) % 10 == 0:
            print("Step %d/%d: T = %.2f, Epsilon = %.2f, Lambda = %.2f" % (t + 1, t_max, T, eps, lmd))

    return W, C
