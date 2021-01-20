import unittest 
import numpy as np 

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


from pycycle.thermo.cea.species_data import janaf
from pycycle.elements.cfd_start import CFDStart
from pycycle.mp_cycle import Cycle


class CFDStartTestCase(unittest.TestCase): 

    def test_case1(self): 

        p = om.Problem()

        p.model = Cycle()
        p.model.options['thermo_method'] = 'CEA'
        p.model.options['thermo_data'] = janaf

        cfd_start = p.model.add_subsystem('cfd_start', CFDStart(), promotes=['*'])

        p.model.set_input_defaults('Ps', units='kPa', val=100)
        p.model.set_input_defaults('V', units='m/s', val=100.)
        p.model.set_input_defaults('area', units='m**2', val=1)
        p.model.set_input_defaults('W', units='kg/s', val=100.)

        p.setup()
        p.run_model()


        tol = 1e-4
        assert_near_equal(p['Fl_O:tot:P'], 15.24202341, tol) # psi
        assert_near_equal(p.get_val('Fl_O:stat:P', units='kPa'), 100, tol)
        assert_near_equal(p.get_val('Fl_O:stat:MN'), 0.26744049, tol)


if __name__ == "__main__": 

    unittest.main()