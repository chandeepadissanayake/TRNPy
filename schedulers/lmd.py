from . import exp_decay


def sch_lmd_static(lmd_i, lmd_f, t, t_max):
    return lmd_i


def sch_lmd_exp_raise(lmd_i, lmd_f, t, t_max):
    return exp_decay(lmd_i, lmd_f, t, t_max)
