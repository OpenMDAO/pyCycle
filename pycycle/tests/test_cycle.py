import unittest

import openmdao.api as om

import pycycle.api as pyc


# very simple cycle, design case only. 
# purpose is specifically to test is the 
# flow-station setup stuff works across sub-groups
class Turbojet(pyc.Cycle):

    def initialize(self): 

        self.options.declare('promoted', default=True)
        super().initialize()

    def setup(self):

        self.options['thermo_method'] = 'TABULAR'
        self.options['thermo_data'] = pyc.AIR_JETA_TAB_SPEC
        FUEL_TYPE = "FAR"


        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())

        # Connect flow stations
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I', connect_w=False)


        promotes=None
        if self.options['promoted']: 
            promotes=['*']
            self.pyc_connect_flow('inlet.Fl_O', 'comp.Fl_I')
        else: 
            promotes=None
            self.pyc_connect_flow('inlet.Fl_O', 'sub_cycle.comp.Fl_I')

        sc = self.add_subsystem('sub_cycle', pyc.Cycle(thermo_method=self.options['thermo_method'], 
                                                       thermo_data=self.options['thermo_data']), 
                                promotes=promotes)
        sc.add_subsystem('comp', pyc.Compressor(map_data=pyc.AXI5, map_extrap=True),
                            promotes_inputs=['Nmech'])
        sc.add_subsystem('burner', pyc.Combustor(fuel_type=FUEL_TYPE))
        sc.add_subsystem('turb', pyc.Turbine(map_data=pyc.LPT2269),
                                    promotes_inputs=['Nmech'])
        sc.add_subsystem('nozz', pyc.Nozzle(nozzType='CD', lossCoef='Cv'))
        sc.add_subsystem('shaft', pyc.Shaft(num_ports=2),promotes_inputs=['Nmech'])


        sc.pyc_connect_flow('comp.Fl_O', 'burner.Fl_I')
        sc.pyc_connect_flow('burner.Fl_O', 'turb.Fl_I')
        sc.pyc_connect_flow('turb.Fl_O', 'nozz.Fl_I')

        # Make other non-flow connections
        # Connect turbomachinery elements to shaft
        sc.connect('comp.trq', 'shaft.trq_0')
        sc.connect('turb.trq', 'shaft.trq_1')

        # Connnect nozzle exhaust to freestream static conditions
        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')

        super().setup()


class TestCycle(unittest.TestCase):
    
    def test_sub_cycle_promoted(self): 

        p = om.Problem()
        p.model = Turbojet(promoted=True)

        p.setup()

    def test_sub_cycle_not_promoted(self): 

        p = om.Problem()
        p.model = Turbojet(promoted=False)

        p.setup()

if __name__ == "__main__": 

    unittest.main()