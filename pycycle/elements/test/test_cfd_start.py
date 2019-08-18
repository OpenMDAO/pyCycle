import unittest 
import numpy as np 

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error


from pycycle.cea.species_data import janaf
from pycycle.elements.cfd_start import CFDStart
from pycycle.constants import AIR_MIX


class CFDStartTestCase(unittest.TestCase): 

    def test_case1(self): 

        p = om.Problem()

        des_vars = p.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])
        des_vars.add_output('Ps', units='kPa', val=100)
        des_vars.add_output('V', units='m/s', val=100.)
        des_vars.add_output('area', units='m**2', val=1)
        des_vars.add_output('W', units='kg/s', val=100.)

        p.model.add_subsystem('cfd_start', CFDStart(), promotes=['*'])

        p.setup()
        p.run_model()


        tol = 1e-4
        assert_rel_error(self, p['Fl_O:tot:P'], 15.24202341, tol) # psi
        assert_rel_error(self, p.get_val('Fl_O:stat:P', units='kPa'), 100, tol)
        assert_rel_error(self, p.get_val('Fl_O:stat:MN'), 0.26744049, tol)


if __name__ == "__main__": 

    unittest.main()