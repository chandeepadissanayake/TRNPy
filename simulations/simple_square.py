import distributions.general as distribs_general
import schedulers.lifetime as sch_lifetime
import schedulers.stepsize as sch_stepsize
import schedulers.lmd as sch_lmd
import simulations.confhelpers.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import trn

""" Configuration """
# Number of pointers
N = 200
# Initial pointer distribution
W_distrib = distribs_general.random
# Maximum number of iterations
t_max = 200 * N
# Lifetime and its scheduler
T_i = 0.1 * N
T_f = 2 * N
# T_sch = sch_lifetime.sch_T_static
T_sch = sch_lifetime.sch_T_exp_raise
# Step Size and its scheduler
eps_i = 0.3
eps_f = 0.05
# eps_sch = sch_stepsize.sch_eps_static
eps_sch = sch_stepsize.sch_eps_exp_raise
# Lambda and its scheduler
lmd_i = 0.2 * N
lmd_f = 0.01
# lmd_sch = sch_lmd.sch_lmd_static
lmd_sch = sch_lmd.sch_lmd_exp_raise


def _prepare_plot(V):
    mins = np.amin(V, axis=0)
    xmin, ymin = mins[0], mins[1]

    maxs = np.amax(V, axis=0)
    xmax, ymax = maxs[0], maxs[1]

    w = xmax - xmin
    h = ymax - ymin

    curr_ax = plt.gca()

    return xmin, ymin, xmax, ymax, w, h, curr_ax


def visualize_manifold(V):
    xmin, ymin, xmax, ymax, w, h, curr_ax = _prepare_plot(V)

    square = patches.Rectangle((xmin, ymin), w, h, linewidth=1, edgecolor="g", facecolor="none")
    curr_ax.add_patch(square)

    curr_ax.scatter(V[:, 0], V[:, 1])

    plt.title("Manifold")


def visualize_init_W(V, W):
    xmin, ymin, xmax, ymax, w, h, curr_ax = _prepare_plot(V)

    square = patches.Rectangle((xmin, ymin), w, h, linewidth=1, edgecolor="g", facecolor="none")
    curr_ax.add_patch(square)
    curr_ax.scatter(W[0, :], W[1, :])

    plt.title("Manifold with Initial Pointers")


def visualize_trn(V, W, C):
    xmin, ymin, xmax, ymax, w, h, curr_ax = _prepare_plot(V)

    square = patches.Rectangle((xmin, ymin), w, h, linewidth=1, edgecolor="g", facecolor="none")
    curr_ax.add_patch(square)

    curr_ax.scatter(W[0, :], W[1, :])

    x1s, y1s = [], []
    x2s, y2s = [], []

    conns = np.argwhere(C > 0)
    for conn in conns:
        i, j = conn[0], conn[1]

        x1, y1 = W[0, i], W[1, i]
        x2, y2 = W[0, j], W[1, j]

        # Checking whether this is a repeated connection due to symmetric nature of the connection strength matrix.
        repeated = False
        for m in range(len(x1s)):
            x1t, y1t = x1s[m], y1s[m]
            x2t, y2t = x2s[m], y2s[m]

            if x1 == x2t and y1 == y2t and x2 == x1t and y2 == y1t:
                repeated = True
                break

        if repeated:
            continue

        x1s.append(x1)
        y1s.append(y1)
        x2s.append(x2)
        y2s.append(y2)

    curr_ax.plot(x1s, y1s, x2s, y2s, marker='o', color='b')

    plt.title("Manifold with Adapted Pointers")


def main():
    # Creating the dataset
    V = datasets.generate_square_manifold(M=200)
    # Visualizing manifold with the help of input patterns.
    visualize_manifold(V)
    plt.show()

    # Setting up pointers using initial distribution
    W = W_distrib(V.shape[1], N)

    # Visualizing manifolds with initial pointers
    visualize_init_W(V, W)
    plt.show()

    # Adaptation
    W_u, C = trn.adapt(W, V.T, t_max,  # V transposed to satisfy (D, ) dimension requirement of V
                       T_i, T_f, T_sch,
                       eps_i, eps_f, eps_sch,
                       lmd_i, lmd_f, lmd_sch)
    # Visualizing manifold with adapted pointers
    visualize_trn(V, W_u, C)
    plt.show()


if __name__ == "__main__":
    main()
