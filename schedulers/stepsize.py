from . import exp_decay


def sch_eps_static(eps_i, eps_f, t, t_max):
    return eps_i


def sch_eps_exp_raise(eps_i, eps_f, t, t_max):
    return exp_decay(eps_i, eps_f, t, t_max)
