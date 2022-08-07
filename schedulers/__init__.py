
def exp_decay(i, f, t, T):
    """
    Decays the initial value in an exponential manner with respective to the time.
    :param i: Initial value
    :param f: Final value
    :param t: Current time step
    :param T: Maximum time steps possible
    :return: Current value after decaying at t time steps.
    """

    return i * ((f / i) ** (t / T))
