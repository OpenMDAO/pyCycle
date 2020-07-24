import os.path
import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.units import convert_units as cu

from pycycle.cea.species_data import janaf, Thermo
from pycycle.cea.set_total import SetTotal
from pycycle.cea.set_static import SetStatic
from pycycle import constants

fpath = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(fpath, 'NPSS_Static_CEA_Data.csv')
ref_data = np.loadtxt(data_path, delimiter=",", skiprows=1)

header = ['W', 'MN', 'V', 'A', 's', 'Pt', 'Tt', 'ht', 'rhot',
          'gamt', 'Ps', 'Ts', 'hs', 'rhos', 'gams']
h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class TestSetStaticPs(unittest.TestCase):

    def test_case_Ps(self):

        thermo = Thermo(janaf, init_reacts=constants.janaf_init_prod_amounts)

        p = Problem()

        # the remaining indeps will be removed when set_input_defaults is fixed
        indeps = p.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        indeps.add_output('W', val=1., units='lbm/s')

        p.model.add_subsystem('set_total_TP', SetTotal(thermo_data=janaf), promotes=['b0'])
        p.model.add_subsystem('set_static_Ps', SetStatic(mode='Ps', thermo_data=janaf), promotes=['b0'])
        p.model.set_input_defaults('b0', thermo.b0)
        p.model.set_input_defaults('set_total_TP.T', val=518., units='degR')
        p.model.set_input_defaults('set_total_TP.P', val=14.7, units='psi')
        p.model.set_input_defaults('set_static_Ps.Ps', val=13.0, units='psi')

        p.model.connect('set_total_TP.flow:S', 'set_static_Ps.S')
        p.model.connect('set_total_TP.flow:h', 'set_static_Ps.ht')
        p.model.connect('W', 'set_static_Ps.W')

        p.setup(check=False)
        p.set_solver_print(level=-1)

        # 4 cases to check against
        for i, data in enumerate(ref_data):

            p['set_total_TP.T'] = data[h_map['Tt']]
            p['set_total_TP.P'] = data[h_map['Pt']]

            p['set_static_Ps.Ps'] = data[h_map['Ps']]
            p['W'] = data[h_map['W']]

            p.run_model()

            # check outputs
            npss_vars = ('Ps', 'Ts', 'MN', 'hs', 'rhos', 'gams', 'V', 'A', 's', 'ht')
            Ps, Ts, MN, hs, rhos, gams, V, A, S, ht = tuple(
                [data[h_map[v_name]] for v_name in npss_vars])

            Ps_computed = p['set_static_Ps.flow:P']
            Ts_computed = p['set_static_Ps.flow:T']
            hs_computed = p['set_static_Ps.flow:h']
            rhos_computed = p['set_static_Ps.flow:rho']
            gams_computed = p['set_static_Ps.flow:gamma']
            V_computed = p['set_static_Ps.flow:V']
            A_computed = p['set_static_Ps.flow:area']
            MN_computed = p['set_static_Ps.flow:MN']

            if MN >= .05:
                tol = 3e-4
            else:
                tol = .2
                # MN values off for low MN cases don't match well, but NPSS doesn't solve well down there

            assert_near_equal(MN_computed, MN, tol)
            assert_near_equal(gams_computed, gams, tol)
            assert_near_equal(Ps_computed, Ps, tol)
            assert_near_equal(Ts_computed, Ts, tol)
            assert_near_equal(hs_computed, hs, tol)
            assert_near_equal(rhos_computed, rhos, tol)
            assert_near_equal(V_computed, V, tol)
            assert_near_equal(A_computed, A, tol)

        p.check_partials(includes=['set_static_Ps.statics.ps_calc'], compact_print=True)


if __name__ == "__main__":
    np.seterr(all='raise')
    unittest.main()
