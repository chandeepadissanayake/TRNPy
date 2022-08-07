from . import exp_decay


def sch_T_static(T_i, T_f, t, t_max):
    return T_i


def sch_T_exp_raise(T_i, T_f, t, t_max):
    return exp_decay(T_i, T_f, t, t_max)
