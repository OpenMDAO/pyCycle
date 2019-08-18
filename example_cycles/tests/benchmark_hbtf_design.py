import numpy as np
import unittest
import os


import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error

from example_cycles.high_bypass_turbofan import HBTF


class CFM56DesignTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = om.Problem()

        des_vars = self.prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])
        des_vars.add_output('alt', 35000., units='ft'),
        des_vars.add_output('MN', 0.8),
        des_vars.add_output('T4max', 2857.0, units='degR'),
        des_vars.add_output('Fn_des', 5500.0, units='lbf'),
        des_vars.add_output('inlet:ram_recovery', 0.9990),
        des_vars.add_output('inlet:MN_out', 0.751),
        des_vars.add_output('fan:PRdes', 1.685),
        des_vars.add_output('fan:effDes', 0.8948),
        des_vars.add_output('fan:MN_out', 0.4578)
        des_vars.add_output('splitter:BPR', 5.105),
        des_vars.add_output('splitter:MN_out1', 0.3104)
        des_vars.add_output('splitter:MN_out2', 0.4518)
        des_vars.add_output('duct4:dPqP', 0.0048),
        des_vars.add_output('duct4:MN_out', 0.3121),
        des_vars.add_output('lpc:PRdes', 1.935),
        des_vars.add_output('lpc:effDes', 0.9243),
        des_vars.add_output('lpc:MN_out', 0.3059),
        des_vars.add_output('duct6:dPqP', 0.0101),
        des_vars.add_output('duct6:MN_out', 0.3563),
        des_vars.add_output('hpc:PRdes', 9.369),
        des_vars.add_output('hpc:effDes', 0.8707),
        des_vars.add_output('hpc:MN_out', 0.2442),
        des_vars.add_output('bld3:MN_out', 0.3000)
        des_vars.add_output('burner:dPqP', 0.0540),
        des_vars.add_output('burner:MN_out', 0.1025),
        des_vars.add_output('hpt:effDes', 0.8888),
        des_vars.add_output('hpt:MN_out', 0.3650),
        des_vars.add_output('duct11:dPqP', 0.0051),
        des_vars.add_output('duct11:MN_out', 0.3063),
        des_vars.add_output('lpt:effDes', 0.8996),
        des_vars.add_output('lpt:MN_out', 0.4127),
        des_vars.add_output('duct13:dPqP', 0.0107),
        des_vars.add_output('duct13:MN_out', 0.4463),
        des_vars.add_output('core_nozz:Cv', 0.9933),
        des_vars.add_output('bypBld:frac_W', 0.005),
        des_vars.add_output('bypBld:MN_out', 0.4489),
        des_vars.add_output('duct15:dPqP', 0.0149),
        des_vars.add_output('duct15:MN_out', 0.4589),
        des_vars.add_output('byp_nozz:Cv', 0.9939),
        des_vars.add_output('lp_shaft:Nmech', 4666.1, units='rpm'),
        des_vars.add_output('hp_shaft:Nmech', 14705.7, units='rpm'),
        des_vars.add_output('hp_shaft:HPX', 250.0, units='hp'),
        des_vars.add_output('hpc:cool1:frac_W', 0.050708),
        des_vars.add_output('hpc:cool1:frac_P', 0.5),
        des_vars.add_output('hpc:cool1:frac_work', 0.5),
        des_vars.add_output('hpc:cool2:frac_W', 0.020274),
        des_vars.add_output('hpc:cool2:frac_P', 0.55),
        des_vars.add_output('hpc:cool2:frac_work', 0.5),
        des_vars.add_output('bld3:cool3:frac_W', 0.067214),
        des_vars.add_output('bld3:cool4:frac_W', 0.101256),
        des_vars.add_output('hpc:cust:frac_W', 0.0445),
        des_vars.add_output('hpc:cust:frac_P', 0.5),
        des_vars.add_output('hpc:cust:frac_work', 0.5),
        des_vars.add_output('hpt:cool3:frac_P', 1.0),
        des_vars.add_output('hpt:cool4:frac_P', 0.0),
        des_vars.add_output('lpt:cool1:frac_P', 1.0),
        des_vars.add_output('lpt:cool2:frac_P', 0.0),

        self.prob.model.add_subsystem('DESIGN', HBTF())

        self.prob.model.connect('alt', 'DESIGN.fc.alt')
        self.prob.model.connect('MN', 'DESIGN.fc.MN')
        self.prob.model.connect('Fn_des', 'DESIGN.balance.rhs:W')
        self.prob.model.connect('T4max', 'DESIGN.balance.rhs:FAR')
        self.prob.model.connect('inlet:ram_recovery', 'DESIGN.inlet.ram_recovery')
        self.prob.model.connect('inlet:MN_out', 'DESIGN.inlet.MN')
        self.prob.model.connect('fan:PRdes', 'DESIGN.fan.PR')
        self.prob.model.connect('fan:effDes', 'DESIGN.fan.eff')
        self.prob.model.connect('fan:MN_out', 'DESIGN.fan.MN')
        self.prob.model.connect('splitter:BPR', 'DESIGN.splitter.BPR')
        self.prob.model.connect('splitter:MN_out1', 'DESIGN.splitter.MN1')
        self.prob.model.connect('splitter:MN_out2', 'DESIGN.splitter.MN2')
        self.prob.model.connect('duct4:dPqP', 'DESIGN.duct4.dPqP')
        self.prob.model.connect('duct4:MN_out', 'DESIGN.duct4.MN')
        self.prob.model.connect('lpc:PRdes', 'DESIGN.lpc.PR')
        self.prob.model.connect('lpc:effDes', 'DESIGN.lpc.eff')
        self.prob.model.connect('lpc:MN_out', 'DESIGN.lpc.MN')
        self.prob.model.connect('duct6:dPqP', 'DESIGN.duct6.dPqP')
        self.prob.model.connect('duct6:MN_out', 'DESIGN.duct6.MN')
        self.prob.model.connect('hpc:PRdes', 'DESIGN.hpc.PR')
        self.prob.model.connect('hpc:effDes', 'DESIGN.hpc.eff')
        self.prob.model.connect('hpc:MN_out', 'DESIGN.hpc.MN')
        self.prob.model.connect('bld3:MN_out', 'DESIGN.bld3.MN')
        self.prob.model.connect('burner:dPqP', 'DESIGN.burner.dPqP')
        self.prob.model.connect('burner:MN_out', 'DESIGN.burner.MN')
        self.prob.model.connect('hpt:effDes', 'DESIGN.hpt.eff')
        self.prob.model.connect('hpt:MN_out', 'DESIGN.hpt.MN')
        self.prob.model.connect('duct11:dPqP', 'DESIGN.duct11.dPqP')
        self.prob.model.connect('duct11:MN_out', 'DESIGN.duct11.MN')
        self.prob.model.connect('lpt:effDes', 'DESIGN.lpt.eff')
        self.prob.model.connect('lpt:MN_out', 'DESIGN.lpt.MN')
        self.prob.model.connect('duct13:dPqP', 'DESIGN.duct13.dPqP')
        self.prob.model.connect('duct13:MN_out', 'DESIGN.duct13.MN')
        self.prob.model.connect('core_nozz:Cv', 'DESIGN.core_nozz.Cv')
        self.prob.model.connect('bypBld:MN_out', 'DESIGN.byp_bld.MN')
        self.prob.model.connect('duct15:dPqP', 'DESIGN.duct15.dPqP')
        self.prob.model.connect('duct15:MN_out', 'DESIGN.duct15.MN')
        self.prob.model.connect('byp_nozz:Cv', 'DESIGN.byp_nozz.Cv')
        self.prob.model.connect('lp_shaft:Nmech', 'DESIGN.LP_Nmech')
        self.prob.model.connect('hp_shaft:Nmech', 'DESIGN.HP_Nmech')
        self.prob.model.connect('hp_shaft:HPX', 'DESIGN.hp_shaft.HPX')
        self.prob.model.connect('hpc:cool1:frac_W', 'DESIGN.hpc.cool1:frac_W')
        self.prob.model.connect('hpc:cool1:frac_P', 'DESIGN.hpc.cool1:frac_P')
        self.prob.model.connect('hpc:cool1:frac_work', 'DESIGN.hpc.cool1:frac_work')
        self.prob.model.connect('hpc:cool2:frac_W', 'DESIGN.hpc.cool2:frac_W')
        self.prob.model.connect('hpc:cool2:frac_P', 'DESIGN.hpc.cool2:frac_P')
        self.prob.model.connect('hpc:cool2:frac_work', 'DESIGN.hpc.cool2:frac_work')
        self.prob.model.connect('bld3:cool3:frac_W', 'DESIGN.bld3.cool3:frac_W')
        self.prob.model.connect('bld3:cool4:frac_W', 'DESIGN.bld3.cool4:frac_W')
        self.prob.model.connect('hpc:cust:frac_W', 'DESIGN.hpc.cust:frac_W')
        self.prob.model.connect('hpc:cust:frac_P', 'DESIGN.hpc.cust:frac_P')
        self.prob.model.connect('hpc:cust:frac_work', 'DESIGN.hpc.cust:frac_work')
        self.prob.model.connect('hpt:cool3:frac_P', 'DESIGN.hpt.cool3:frac_P')
        self.prob.model.connect('hpt:cool4:frac_P', 'DESIGN.hpt.cool4:frac_P')
        self.prob.model.connect('lpt:cool1:frac_P', 'DESIGN.lpt.cool1:frac_P')
        self.prob.model.connect('lpt:cool2:frac_P', 'DESIGN.lpt.cool2:frac_P')
        self.prob.model.connect('bypBld:frac_W', 'DESIGN.byp_bld.bypBld:frac_W')

        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.setup(check=False)

        self.prob['DESIGN.balance.FAR'] = 0.025
        self.prob['DESIGN.balance.W'] = 316.0
        self.prob['DESIGN.balance.lpt_PR'] = 4.4
        self.prob['DESIGN.balance.hpt_PR'] = 3.6
        self.prob['DESIGN.fc.balance.Pt'] = 5.2
        self.prob['DESIGN.fc.balance.Tt'] = 440.0

        # from openmdao.api import view_model
        # view_model(self.prob)
        # exit()

    def benchmark_case1(self):
        np.seterr(divide='raise')

        self.prob.run_model()
        tol = 1e-3
        print()

        reg_data = 321.253
        pyc = self.prob['DESIGN.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 30.094
        pyc = self.prob['DESIGN.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.02491
        pyc = self.prob['DESIGN.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 3.6228
        pyc = self.prob['DESIGN.balance.hpt_PR'][0]
        print('HPT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 4.3687
        pyc = self.prob['DESIGN.balance.lpt_PR'][0]
        print('LPT PR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 13274.4
        pyc = self.prob['DESIGN.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.63101
        pyc = self.prob['DESIGN.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1276.48
        pyc = self.prob['DESIGN.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()

if __name__ == "__main__":
    unittest.main()
