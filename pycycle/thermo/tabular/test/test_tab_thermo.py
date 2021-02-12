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

        p['FAR'] = 0.04
        p['P'] = 101325*3
        p['T'] = 1000

        p.run_model()

        TOL = 1e-5
        assert_near_equal(p.get_val('h'), -940543.7934515374, tolerance=TOL)
        assert_near_equal(p.get_val('S'),  7970.880970925528, tolerance=TOL)
        assert_near_equal(p.get_val('gamma'),  1.3095428807991376, tolerance=TOL)
        assert_near_equal(p.get_val('Cp'),  1214.2666696079048, tolerance=TOL)
        assert_near_equal(p.get_val('Cv'),  927.2729246317427, tolerance=TOL)
        assert_near_equal(p.get_val('rho'),  1.0611566773957284, tolerance=TOL)
        assert_near_equal(p.get_val('R'),  286.9948147750743, tolerance=TOL)


if __name__ == "__main__": 

    unittest.main()