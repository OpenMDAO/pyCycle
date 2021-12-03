from collections import namedtuple

import warnings

import openmdao.api as om 
import networkx as nx

from pycycle.element_base import Element
from pycycle.thermo.cea import species_data
from pycycle.constants import ALLOWED_THERMOS


class Cycle(om.Group): 


    def initialize(self):
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')
        self.options.declare('thermo_method', values=ALLOWED_THERMOS, default='CEA',
                              desc='Method for computing thermodynamic properties')

        self.options.declare('thermo_data', default=species_data.janaf,
                              desc='thermodynamic data set.', 
                              recordable=False)

        self._elements = set()

        self._flow_graph = nx.DiGraph()

        # flag needed for user focused error checking to make sure they called super in their sub-class
        self._base_class_super_called = False

        self._children = {}

    def _setup_check(self): 

        if not self._base_class_super_called: 
            raise NotImplementedError(f"`super.setup()` has not been called within the setup method of f{self.__class__}")
        
    def pyc_add_element(self, name, element, **kwargs):
        """
        A thin wrapper around `add_subsystem` to keep track of 
        the elements in a given cycle, separate from the general 
        components (e.g. BalanceComp, ExecComp, etc.)
        """

        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(f"Deprecation warning: `pyc_add_element` function is deprecated because it is no longer needed. " 
                       "Use the `add_subsystem` method." )
        warnings.simplefilter('ignore', DeprecationWarning)

        self.add_subsystem(name, element, **kwargs)


    def add_subsystem(self, name, subsys, **kwargs):
        """
        Customized version of the OpenMDAO Group API method that does 
        additional tracking of elements for the Cycle
        """

        self._children[name] = subsys

        if isinstance(subsys, Element): 
            self._elements.add(subsys)
            if 'thermo_method' in subsys.options:
                subsys.options['thermo_method'] = self.options['thermo_method']

            self._flow_graph.add_node(name, type='element')

        #TODO: Find some way to error check based on _base_class_super_called to let user know they forgot a call to super
        return super().add_subsystem(name, subsys, **kwargs)


    def setup(self): 

        self._base_class_super_called = True


        # Code that follows the flow-graph and propagates thermo setup data down the chain
        node_types = nx.get_node_attributes(self._flow_graph, 'type')
        node_parents = nx.get_node_attributes(self._flow_graph, 'parent')
        node_port_names = nx.get_node_attributes(self._flow_graph, 'port_name')

        G = self._flow_graph

        # put all starting nodes into a FIFO queue
        queue = [node for node in self._flow_graph if G.in_degree(node) == 0]
        visited = set() # use a set, because checking "in" on a queue is slow


        # loop over all child subsystems and push down cycle level options 
        cycle_level_options = ['thermo_method', 'thermo_data', 'design']
        for child_name, child in self._children.items():
            for opt in cycle_level_options: 
                if opt in child.options: 
                    child.options[opt] = self.options[opt]


        # note: three kinds of nodes in graph, elements, in_ports, out_ports. 
        #       The graph is a tree with (potentially) multiple separate root nodes. 
        #       We will do a breadth first search, starting from all starting nodes at the same time. 
        #       This makes sure that Elements with multiple inputs will have all
        #       predecessors set up before we get to them. 
        while queue:
            node = queue.pop(0) 
            node_type = node_types[node]

            # make sure we've already processed all predecessor nodes
            # if not skip this one, we'll hit it again later
            ready_for_node = True

            for p in G.predecessors(node): 
                if p not in visited: 
                    ready_for_node = False
                    break
            if not ready_for_node:
                continue

            queue.extend(G.successors(node))

            if node not in visited: 

                if node_type == 'element': 
                    node_element = self._get_subsystem(node)
                    node_element.pyc_setup_output_ports()

                # connection will be out_port -> in_port
                elif node_type == 'out_port': 
                    src_element = self._get_subsystem(node_parents[node])
                    links = G.out_edges(node)
                    for link in links: 
                        # in almost every case there should only be one link, because otherwise you are creating extra mass flow 
                        # the one exception is for the cooling calcs, which get some "weak" connections from turbine and bleed srcs
                        
                        target_element = self._get_subsystem(node_parents[link[1]])

                        # if target element is None there are two options: 
                        # 1) there is a sub-cycle that you need to push into 
                        #        in this case, get the containing sub-cycle and push some starting nodes into its graph based on this linkage
                        # 2) they made a mistake in the element name, so throw an error
                        # print(node_parents[link[1]])

                        out_port = node_port_names[node]
                        in_port = node_port_names[link[1]]
                        # this passes whatever configuration data there was from the src element to the target keyed by port names

                        if out_port not in src_element.Fl_O_data: 
                            raise RuntimeError(f'in {self.pathname},{src_element.pathname}.{out_port} has not been properly setup.'
                                               f'something is wrong with one of your `pyc_setup_output_ports` method in {src_element.pathname}')

                        target_element.Fl_I_data[in_port] = src_element.Fl_O_data[out_port]

                visited.add(node)


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

        # build the directed graph of flow connections
        src_element_name = ''.join(fl_src.split('.')[:-1])
        src_port_name = fl_src.split('.')[-1]

        target_elment_name = ''.join(fl_target.split('.')[:-1])
        target_port_name = fl_target.split('.')[-1]

        # element nodes are needed so we can map from flow ports through elements
        # self._flow_graph.add_node(src_element_name, type='element')
        # self._flow_graph.add_node(target_elment_name, type='element')
        
        self._flow_graph.add_node(fl_src, type='out_port', parent=src_element_name, port_name=src_port_name)
        self._flow_graph.add_node(fl_target, type='in_port', parent=target_elment_name, port_name=target_port_name)

        self._flow_graph.add_edge(src_element_name, fl_src)
        self._flow_graph.add_edge(fl_src, fl_target)
        self._flow_graph.add_edge(fl_target, target_elment_name)

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


