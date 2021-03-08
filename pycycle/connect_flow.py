import warnings


def connect_flow(group, fl_src, fl_target, connect_stat=True, connect_tot=True, connect_w=True):
    """ connect flow variables """

    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn(f"Deprecation warning: `connect_flow` function is deprecated. Use the `pyc_connect_flow` method from `Cycle` class instead." )
    warnings.simplefilter('ignore', DeprecationWarning)

    group.pyc_connect_flow(fl_src, fl_target, connect_stat, connect_tot, connect_w)