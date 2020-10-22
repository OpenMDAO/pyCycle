import unittest

import numpy as np

import openmdao.api as om

from pycycle.thermo.cea import species_data
from pycycle.constants import AIR_ELEMENTS
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.elements.mix_ratio import MixRatio

class MixRatioTestCase(unittest.TestCase):
    def test_mix_fuel(self): 

            thermo_spec = species_data.janaf

            air_thermo = species_data.Properties(thermo_spec, init_elements=AIR_ELEMENTS)

            p = om.Problem()

            fuel = 'JP-7'
            p.model = MixRatio(inflow_thermo_data=thermo_spec, thermo_data=thermo_spec,
                               inflow_elements=AIR_ELEMENTS, mix_reactant=fuel)


            p.setup(force_alloc_complex=True)

            # p['Fl_I:stat:P'] = 158.428
            p['Fl_I:stat:W'] = 38.8
            p['mix_ratio'] = 0.02673
            p['Fl_I:tot:h'] = 181.381769
            p['reactant_Tt'] = 518.
            p['Fl_I:tot:b0'] = air_thermo.b0

            p.run_model()

            tol = 5e-7
            assert_near_equal(p['mass_avg_h'], 176.65965638, tolerance=tol)
            assert_near_equal(p['Wout'], 39.837124, tolerance=tol)
            assert_near_equal(p['b0_out'], np.array([0.0003149, 0.00186566, 0.00371394, 0.05251212, 0.01410888]), tolerance=tol)

            data = p.check_partials(out_stream=None, method='cs')
            # data = p.check_partials(method='cs')
            assert_check_partials(data, atol=1.e-6, rtol=1.e-6)


if __name__ == "__main__": 

    unittest.main()