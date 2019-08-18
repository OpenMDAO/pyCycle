def connect_flow(group, fl_src, fl_target, connect_stat=True, connect_tot=True, connect_w=True):
    """ connect flow variables """

    # total
    if connect_tot:
        for v_name in ('h','T','P','S','rho','gamma','Cp','Cv','n', 'R'):
            group.connect('%s:tot:%s'%(fl_src, v_name), '%s:tot:%s'%(fl_target, v_name))

    # static
    if connect_stat:
        for v_name in ('V', 'Vsonic'):  # ('Wc', 'W', 'FAR'):
            group.connect('%s:stat:%s'%(fl_src, v_name), '%s:stat:%s'%(fl_target, v_name))

        for v_name in ('Cp', 'Cv', 'MN', 'P', 'S', 'T', 'area', 'gamma', 'h', 'n', 'rho'):
            group.connect('%s:stat:%s'%(fl_src, v_name), '%s:stat:%s'%(fl_target, v_name))

    if connect_w:
       group.connect('%s:stat:W'%(fl_src,), '%s:stat:W'%(fl_target,))