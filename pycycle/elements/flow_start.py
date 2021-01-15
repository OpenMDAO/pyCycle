import numpy as np

from openmdao.api import Group, ExplicitComponent

from pycycle.thermo.cea import species_data
from pycycle.thermo.thermo import Thermo, ThermoAdd
from pycycle.constants import CEA_AIR_COMPOSITION, CEA_WET_AIR_COMPOSITION
from pycycle.element_base import Element


class FlowStart(Element):

    def initialize(self):
        self.options.declare('elements', default=CEA_AIR_COMPOSITION,
                              desc='set of elements present in the flow')

        self.options.declare('use_WAR', default=False, values=[True, False], 
                              desc='If True, includes WAR calculation')

        super().initialize()

    def pyc_setup_output_ports(self): 
        elements = self.options['elements']
        
        self.init_output_flow('Fl_O', elements)

    def setup(self):
        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        use_WAR = self.options['use_WAR']

        elements = self.Fl_O_data['Fl_O']

        if use_WAR == True:
            if 'H' not in elements or 'O' not in elements:
                raise ValueError('The provided elements to FlightConditions do not contain H or O. In order to specify a nonzero WAR the elements must contain both H and O.')

        # inputs
        if use_WAR == True:

            mix = ThermoAdd(method=thermo_method, thermo_kwargs={'spec':thermo_data, 
                                                                 'inflow_elements':elements, 
                                                                 'mix_elements':'Water', })
            self.add_subsystem('WAR', mix, 
                                promotes_inputs=(('Fl_I:stat:W', 'W'), ('mix:ratio', 'WAR')), 
                                promotes_outputs=(('composition_out', 'composition'), ))
        

        set_TP = Thermo(mode='total_TP', fl_name='Fl_O:tot', 
                        method=thermo_method, 
                        thermo_kwargs={'elements':elements, 
                                       'spec':thermo_data})

        in_vars = ('T','P', 'composition')

        self.add_subsystem('totals', set_TP, promotes_inputs=in_vars,
                           promotes_outputs=('Fl_O:tot:*',))

        set_stat_MN = Thermo(mode='static_MN', fl_name='Fl_O:stat', 
                             method=thermo_method, 
                             thermo_kwargs={'elements':elements, 
                                            'spec':thermo_data} )

        self.add_subsystem('exit_static', set_stat_MN, promotes_inputs=('MN', 'W', 'composition'),
                           promotes_outputs=('Fl_O:stat:*', ))

        self.connect('totals.h','exit_static.ht')
        self.connect('totals.S','exit_static.S')
        self.connect('Fl_O:tot:P','exit_static.guess:Pt')
        self.connect('totals.gamma', 'exit_static.guess:gamt')

        super().setup()


