import numpy as np

from openmdao.api import Group, ExplicitComponent

from pycycle.thermo.cea import species_data
from pycycle.thermo.thermo import Thermo, ThermoAdd
from pycycle.constants import THERMO_DEFAULT_COMPOSITIONS
from pycycle.element_base import Element


class FlowStart(Element):

    def initialize(self):

        self.options.declare('composition', default=None,
                              desc='composition of the flow. None means using the default for the thermo package')

        self.options.declare('reactant', default=False, types=(bool, str), 
                              desc='If False, flow matches base composition. If a string, then that reactant '
                                   'is mixed into the flow at at the ratio set by the `mix_ratio` input')

        self.options.declare('mix_ratio_name', default='mix:ratio', 
                             desc='The name of the input that governs the mix ratio of the reactant to the primary flow')

        super().initialize()

    def pyc_setup_output_ports(self): 

        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        composition = self.options['composition']
        reactant = self.options['reactant']

        if reactant is not False: 
            self.thermo_add = ThermoAdd(method=thermo_method, mix_mode='reactant', 
                                        thermo_kwargs={'spec':thermo_data, 
                                                       'inflow_composition':composition, 
                                                       'mix_composition':reactant, })
          
            self.init_output_flow('Fl_O', self.thermo_add)

        else: 
            if composition is None: 
                composition = THERMO_DEFAULT_COMPOSITIONS[thermo_method]
            self.init_output_flow('Fl_O', composition)


    def setup(self):
        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        reactant = self.options['reactant']

        composition = self.Fl_O_data['Fl_O']

       
        # inputs
        if reactant is not False :
            mix_ratio_name = self.options['mix_ratio_name']

            
            self.add_subsystem('thermo_add', self.thermo_add, 
                                promotes_inputs=(('Fl_I:stat:W', 'W'), ('mix:ratio', mix_ratio_name)), 
                                promotes_outputs=(('composition_out', 'composition'), ))
        
        set_TP = Thermo(mode='total_TP', fl_name='Fl_O:tot', 
                        method=thermo_method, 
                        thermo_kwargs={'composition':composition, 
                                       'spec':thermo_data})

        in_vars = ('T','P', 'composition')

        self.add_subsystem('totals', set_TP, promotes_inputs=in_vars,
                           promotes_outputs=('Fl_O:tot:*',))

        set_stat_MN = Thermo(mode='static_MN', fl_name='Fl_O:stat', 
                             method=thermo_method, 
                             thermo_kwargs={'composition':composition, 
                                            'spec':thermo_data} )

        self.add_subsystem('exit_static', set_stat_MN, promotes_inputs=('MN', 'W', 'composition'),
                           promotes_outputs=('Fl_O:stat:*', ))

        self.connect('totals.h','exit_static.ht')
        self.connect('totals.S','exit_static.S')
        self.connect('Fl_O:tot:P','exit_static.guess:Pt')
        self.connect('totals.gamma', 'exit_static.guess:gamt')

        super().setup()


