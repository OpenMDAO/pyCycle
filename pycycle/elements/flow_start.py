import numpy as np

from openmdao.api import Group, ExplicitComponent

from pycycle.thermo.cea import species_data
from pycycle.thermo.thermo import Thermo
from pycycle.elements.mix_ratio import MixRatio
from pycycle.constants import AIR_ELEMENTS, WET_AIR_ELEMENTS


class FlowStart(Group):

    def initialize(self):

        self.options.declare('thermo_data', default=species_data.janaf,
                              desc='thermodynamic data set', recordable=False)
        self.options.declare('elements', default=AIR_ELEMENTS,
                              desc='set of elements present in the flow')

        self.options.declare('use_WAR', default=False, values=[True, False], 
                              desc='If True, includes WAR calculation')

    def setup(self):
        thermo_data = self.options['thermo_data']
        elements = self.options['elements']
        use_WAR = self.options['use_WAR']

        if use_WAR == True:
            if 'H' not in elements or 'O' not in elements:
                raise ValueError('The provided elements to FlightConditions do not contain H or O. In order to specify a nonzero WAR the elements must contain both H and O.')

        elif use_WAR == False:
            if 'H' in elements.keys():

                raise ValueError('In order to provide elements containing H, a nonzero water to air ratio (WAR) must be specified. Set the option use_WAR to True and give a non zero WAR.')

        thermo = species_data.Properties(thermo_data, init_elements=elements)
        self.air_prods = thermo.products
        self.num_prod = len(self.air_prods)

        # inputs
        if use_WAR == True:
            mix = MixRatio(inflow_thermo_data=thermo_data, thermo_data=thermo_data,
                           inflow_elements=elements, mix_reactant='water')
            self.add_subsystem('WAR', mix, 
                                promotes_inputs=('Fl_I:tot:b0', 'Fl_I:stat:W', ('mix_ratio', 'WAR')), 
                                promotes_outputs=( ('b0_out', 'b0'), ))
        

        set_TP = Thermo(mode='total_TP', fl_name='Fl_O:tot', 
                        method='CEA', 
                        thermo_kwargs={'elements':elements, 
                                       'spec':thermo_data})

        params = ('T','P', 'b0')

        self.add_subsystem('totals', set_TP, promotes_inputs=params,
                           promotes_outputs=('Fl_O:tot:*',))

        set_stat_MN = Thermo(mode='static_MN', fl_name='Fl_O:stat', 
                             method='CEA', 
                             thermo_kwargs={'elements':elements, 
                                            'spec':thermo_data} )

        self.add_subsystem('exit_static', set_stat_MN, promotes_inputs=('MN', 'W', 'b0'),
                           promotes_outputs=('Fl_O:stat:*', ))

        self.connect('totals.h','exit_static.ht')
        self.connect('totals.S','exit_static.S')
        self.connect('Fl_O:tot:P','exit_static.guess:Pt')
        self.connect('totals.gamma', 'exit_static.guess:gamt')

        self.set_input_defaults('b0', thermo.b0)
