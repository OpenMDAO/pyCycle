import unittest

import numpy as np 

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.constants import AIR_JETA_TAB_SPEC, TAB_AIR_FUEL_COMPOSITION
from pycycle.thermo.tabular.tabular_thermo import SetTotalTP

class TabThermoUnitTest(unittest.TestCase): 

    def test_tab_thermo(self): 

        p = om.Problem()

        p.model = SetTotalTP(spec=AIR_JETA_TAB_SPEC, composition=TAB_AIR_FUEL_COMPOSITION)

        p.setup()

        p['composition'] = 0.04
        p['P'] = 101325*3
        p['T'] = 1000

        p.run_model()

        TOL = 5e-4 
        assert_near_equal(p.get_val('h'), -940746.85004758, tolerance=TOL)
        assert_near_equal(p.get_val('S'),  7967.73852287, tolerance=TOL)
        assert_near_equal(p.get_val('gamma'),  1.3094714, tolerance=TOL)
        assert_near_equal(p.get_val('Cp'),  1214.3836316, tolerance=TOL)
        assert_near_equal(p.get_val('Cv'),  927.3899091, tolerance=TOL)
        assert_near_equal(p.get_val('rho'),  1.05951464, tolerance=TOL)
        assert_near_equal(p.get_val('R'),  286.9948147750743, tolerance=TOL)


if __name__ == "__main__": 

    unittest.main()