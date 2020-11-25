from collections import namedtuple

import openmdao.api as om 

# CycleParam = namedtuple('CycleParam', name, value, units)


class Cycle(om.Group): 

    def __init__(self, **kwargs): 
        """
        A custom group used to model a specific thermodynamic cycle
        """

        self._elements = set()

        super().__init__(**kwargs)

    #     # bit of a hack to get around weird timing in OM's option system
    #     design_default=True
    #     if 'design' in kwargs: 
    #         design_default = kwargs['design']

    #     self.options.declare('design', default=design_default,
    #                           desc='Switch between on-design and off-design calculation.')


    def pyc_add_element(self, name, element,**kwargs):
        """
        A thin wrapper around `add_subsystem` to keep track of 
        the elements in a given cycle, separate from the general 
        components (e.g. BalanceComp, ExecComp, etc.)
        """
        self._elements.add(element)
        self.add_subsystem(name, element, **kwargs)

    def pyc_connect_flow(self, fl_src, fl_target, connect_stat=True, connect_tot=True, connect_w=True):
        """ 
        helper function to connect all of the flow variables between two ports 
        """

        # always connect compositions, because these are shape_by_conn=True
        self.connect(f'{fl_src}:tot:composition', [f'{fl_target}:tot:composition', f'{fl_target}:stat:composition'])
        # total
        if connect_tot:
            for v_name in ('h','T','P','S','rho','gamma','Cp','Cv', 'R'):
                self.connect('%s:tot:%s'%(fl_src, v_name), '%s:tot:%s'%(fl_target, v_name))

        # static
        if connect_stat:
            for v_name in ('V', 'Vsonic'):  # ('Wc', 'W', 'FAR'):
                self.connect('%s:stat:%s'%(fl_src, v_name), '%s:stat:%s'%(fl_target, v_name))

            for v_name in ('Cp', 'Cv', 'MN', 'P', 'S', 'T', 'area', 'gamma', 'h', 'rho'):
                self.connect('%s:stat:%s'%(fl_src, v_name), '%s:stat:%s'%(fl_target, v_name))

        if connect_w:
           self.connect('%s:stat:W'%(fl_src,), '%s:stat:W'%(fl_target,))


class MPCycle(om.Group): 

    def __init__(self, **kwargs): 
        self._cycle_params = {}
        self._des_pnt = None
        self._od_pnts= []
        self._des_od_connections = []
        self._use_default_des_od_conns = False
        super(MPCycle, self).__init__(**kwargs)

    def pyc_add_cycle_param(self, name, val, units=None): 

        # TODO: Throw error if this is called after setup

        if name in self._cycle_params: 
            raise ValueError(f'A cycle parameter named `{name}` already exits.')

        self._cycle_params[name] = (val, units)

    def pyc_connect_des_od(self, src, target): 
        if self._des_pnt is None:
            raise ValueError('Cannot connect between design and off design because no design point has been created. Use pyc_add_pnt to add a design point.')

        elif self._od_pnts == []:
            raise ValueError('Cannot connect between design and off design because no off design point has been created. Use pyc_add_pnt to add an off design point.')

        self._des_od_connections.append((src, target))

    def pyc_use_default_des_od_conns(self, skip=None): 
        if self._des_pnt is None:
            raise ValueError('Cannot connect between design and off design because no design point has been created. Use pyc_add_pnt to add a design point.')

        elif self._od_pnts == []:
            raise ValueError('Cannot connect between design and off design because no off design point has been created. Use pyc_add_pnt to add an off design point.')

        self._default_des_od_cons_skip = skip
        self._use_default_des_od_conns = True

    def pyc_add_pnt(self, name, pnt, **kwargs):
        if pnt.options['design'] is True:
            if self._des_pnt is not None:
                raise ValueError(f'Only one design point is allowed. A design point named `{self._des_pnt.name}` already exists.')

            self.add_subsystem(name, pnt, **kwargs)
            self._des_pnt = pnt
        elif pnt.options['design'] is False:
            self.add_subsystem(name, pnt, **kwargs)
            self._od_pnts.append(pnt)
            
        return pnt

    def configure(self): 
        # after all child pts have been set up, 
        # promote any cycle parameters to this level and set their default values
        # then issue connections between the design and off-design points

        for param, (val, units) in self._cycle_params.items(): 
            self.set_input_defaults(name=param, val=val, units=units)
        
            self.promotes(self._des_pnt.name, inputs=[param])
            for pnt in self._od_pnts: 
                self.promotes(pnt.name, inputs=[param])


        for src, target in self._des_od_connections: 
            for od_pnt in self._od_pnts: 
                self.connect(f'{self._des_pnt.name}.{src}', f'{od_pnt.name}.{target}')
        
        if self._use_default_des_od_conns: 
            skip = self._default_des_od_cons_skip
            for elem in self._des_pnt._elements: 
                if  skip is not None and elem.name in skip: 
                    continue
                try: 
                    for src, target in elem.default_des_od_conns: 
                        for od_pnt in self._od_pnts: 
                            self.connect( f'{self._des_pnt.name}.{elem.name}.{src}', f'{od_pnt.name}.{elem.name}.{target}')
                except AttributeError: 
                    pass # no des-to-od conns defined





