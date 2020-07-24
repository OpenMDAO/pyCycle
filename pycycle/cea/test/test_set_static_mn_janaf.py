import os.path
import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp
from openmdao.utils.assert_utils import assert_near_equal

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


class TestSetStaticMN(unittest.TestCase):

    def test_case_MN(self):

        thermo = Thermo(janaf, init_reacts=constants.janaf_init_prod_amounts)

        p = Problem()

        # the remaining indeps will be removed when set_input_defaults is fixed
        indeps = p.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        indeps.add_output('W', val=1.5, units='lbm/s')

        p.model.add_subsystem('set_total_TP', SetTotal(thermo_data=janaf), promotes=['b0', 'P'])
        p.model.add_subsystem('set_static_MN', SetStatic(mode='MN', thermo_data=janaf), promotes=['b0', ('guess:Pt', 'P')])
        p.model.set_input_defaults('b0', thermo.b0)
        p.model.set_input_defaults('set_static_MN.MN', val=1.5, units=None)
        p.model.set_input_defaults('P', val=14.7, units='psi')
        p.model.set_input_defaults('set_total_TP.T', val=518., units='degR')

        p.model.connect('set_total_TP.flow:S', 'set_static_MN.S')
        p.model.connect('set_total_TP.flow:h', 'set_static_MN.ht')
        p.model.connect('set_total_TP.flow:gamma', 'set_static_MN.guess:gamt')
        p.model.connect('W', 'set_static_MN.W')

        p.set_solver_print(level=-1)
        p.setup(check=False)

        # 4 cases to check against
        for i, data in enumerate(ref_data):

            p['set_total_TP.T'] = data[h_map['Tt']]
            p['P'] = data[h_map['Pt']]

            p['set_static_MN.MN'] = data[h_map['MN']]
            p['W'] = data[h_map['W']]

            p.run_model()

            # check outputs
            npss_vars = ('Ps', 'Ts', 'MN', 'hs', 'rhos', 'gams', 'V', 'A', 's', 'ht')
            Ps, Ts, MN, hs, rhos, gams, V, A, S, ht = tuple(
                [data[h_map[v_name]] for v_name in npss_vars])

            Ps_computed = p['set_static_MN.flow:P']
            Ts_computed = p['set_static_MN.flow:T']
            hs_computed = p['set_static_MN.flow:h']
            rhos_computed = p['set_static_MN.flow:rho']
            gams_computed = p['set_static_MN.flow:gamma']
            V_computed = p['set_static_MN.flow:V']
            A_computed = p['set_static_MN.flow:area']
            MN_computed = p['set_static_MN.flow:MN']

            if MN < 2:
                tol = 1e-4
            else:  # The MN 2.0 case doesn't get as close
                tol = 1e-2

            assert_near_equal(MN_computed, MN, tol)
            assert_near_equal(gams_computed, gams, tol)
            assert_near_equal(Ps_computed, Ps, tol)
            assert_near_equal(Ts_computed, Ts, tol)
            assert_near_equal(hs_computed, hs, tol)
            assert_near_equal(rhos_computed, rhos, tol)
            assert_near_equal(V_computed, V, tol)
            assert_near_equal(A_computed, A, tol)

if __name__ == "__main__":
    import scipy

    np.seterr(all='raise')
    unittest.main()
