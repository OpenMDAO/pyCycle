import numpy as np
import unittest
import os

from openmdao.api import Problem, IndepVarComp
from openmdao.utils.units import convert_units as cu
from openmdao.utils.assert_utils import assert_rel_error

from example_cycles.CFM56 import CFM56


class CFM56OffdesignTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = Problem()

        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        des_vars.add_output('OD_MN', 0.8),
        des_vars.add_output('OD_alt', 35000.0, units='ft'),
        des_vars.add_output('OD_Fn_target', 5500.0, units='lbf'),  # 8950.0
        des_vars.add_output('OD_dTs', 0.0, units='degR')
        des_vars.add_output('OD_cust_fracW', 0.0445)
        # des_vars.add_output('alt', 35000., units='ft'),
        # des_vars.add_output('MN', 0.8),
        # des_vars.add_output('T4max', 2857.0, units='degR'),
        # des_vars.add_output('Fn_des', 5500.0, units='lbf'),
        des_vars.add_output('inlet:ram_recovery', 0.9990),
        des_vars.add_output('duct4:dPqP', 0.0048),
        des_vars.add_output('duct6:dPqP', 0.0101),
        des_vars.add_output('burner:dPqP', 0.0540),
        des_vars.add_output('duct11:dPqP', 0.0051),
        des_vars.add_output('duct13:dPqP', 0.0107),
        des_vars.add_output('core_nozz:Cv', 0.9933),
        des_vars.add_output('bypBld:frac_W', 0.005),
        des_vars.add_output('duct15:dPqP', 0.0149),
        des_vars.add_output('byp_nozz:Cv', 0.9939),
        des_vars.add_output('hp_shaft:HPX', 250.0, units='hp'),
        des_vars.add_output('hpc:cool1:frac_W', 0.050708),
        des_vars.add_output('hpc:cool1:frac_P', 0.5),
        des_vars.add_output('hpc:cool1:frac_work', 0.5),
        des_vars.add_output('hpc:cool2:frac_W', 0.020274),
        des_vars.add_output('hpc:cool2:frac_P', 0.55),
        des_vars.add_output('hpc:cool2:frac_work', 0.5),
        des_vars.add_output('bld3:cool3:frac_W', 0.067214),
        des_vars.add_output('bld3:cool4:frac_W', 0.101256),
        # des_vars.add_output('hpc:cust:frac_W', 0.0445),
        des_vars.add_output('hpc:cust:frac_P', 0.5),
        des_vars.add_output('hpc:cust:frac_work', 0.5),
        des_vars.add_output('hpt:cool3:frac_P', 1.0),
        des_vars.add_output('hpt:cool4:frac_P', 0.0),
        des_vars.add_output('lpt:cool1:frac_P', 1.0),
        des_vars.add_output('lpt:cool2:frac_P', 0.0),

        des_vars.add_output('fan:s_PRdes', 0.999912416431),
        des_vars.add_output('fan:s_WcDes', 1.03246959658),
        des_vars.add_output('fan:s_effDes', 1.00013412617),
        des_vars.add_output('fan:s_NcDes', 5091.84571411),
        des_vars.add_output('lpc:s_PRdes', 1.0),
        des_vars.add_output('lpc:s_WcDes', 1.00411122011),
        des_vars.add_output('lpc:s_effDes', 0.999972953236),
        des_vars.add_output('lpc:s_NcDes', 4640.80978341),
        des_vars.add_output('hpc:s_PRdes', 0.999352552331),
        des_vars.add_output('hpc:s_WcDes', 1.02817130922),
        des_vars.add_output('hpc:s_effDes', 1.00007580683),
        des_vars.add_output('hpc:s_NcDes', 13544.2035253),
        des_vars.add_output('hpt:s_PRdes', 0.524557693866),
        des_vars.add_output('hpt:s_WpDes', 1.39329803688),
        des_vars.add_output('hpt:s_effDes', 0.987775061125),
        des_vars.add_output('hpt:s_NpDes', 2.75125333383),
        des_vars.add_output('lpt:s_PRdes', 0.673736258118),
        des_vars.add_output('lpt:s_WpDes', 1.48034371393),
        des_vars.add_output('lpt:s_effDes', 0.974542303109),
        des_vars.add_output('lpt:s_NpDes', 1.03027097635),

        des_vars.add_output('core_nozz:Throat:stat:area', 397.755002537, units='inch**2')
        des_vars.add_output('byp_nozz:Throat:stat:area', 1316.25610748, units='inch**2')

        des_vars.add_output('inlet:Fl_O:stat:area', 2566.76100868, units='inch**2')
        des_vars.add_output('fan:Fl_O:stat:area', 2228.37737592, units='inch**2')
        des_vars.add_output('splitter:Fl_O1:stat:area', 504.011122272, units='inch**2')
        des_vars.add_output('splitter:Fl_O2:stat:area', 1882.18932965, units='inch**2')
        des_vars.add_output('duct4:Fl_O:stat:area', 503.997116848, units='inch**2')
        des_vars.add_output('lpc:Fl_O:stat:area', 293.579637404, units='inch**2')
        des_vars.add_output('duct6:Fl_O:stat:area', 259.649232657, units='inch**2')
        des_vars.add_output('hpc:Fl_O:stat:area', 49.0540725574, units='inch**2')
        des_vars.add_output('bld3:Fl_O:stat:area', 33.7913500831, units='inch**2')
        des_vars.add_output('burner:Fl_O:stat:area', 157.7233536, units='inch**2')
        des_vars.add_output('hpt:Fl_O:stat:area', 172.74350706, units='inch**2')
        des_vars.add_output('duct11:Fl_O:stat:area', 202.354676631, units='inch**2')
        des_vars.add_output('lpt:Fl_O:stat:area', 613.494988147, units='inch**2')
        des_vars.add_output('duct13:Fl_O:stat:area', 582.849448774, units='inch**2')
        des_vars.add_output('byp_bld:Fl_O:stat:area', 1882.04141644, units='inch**2')
        des_vars.add_output('duct15:Fl_O:stat:area', 1878.67377328, units='inch**2')

        self.prob.model.add_subsystem('OD', CFM56(design=False, statics=True))

        self.prob.model.connect('OD_alt', 'OD.fc.alt')
        self.prob.model.connect('OD_MN', 'OD.fc.MN')
        self.prob.model.connect('OD_Fn_target', 'OD.balance.rhs:FAR')
        self.prob.model.connect('OD_dTs', 'OD.fc.dTs')
        self.prob.model.connect('OD_cust_fracW', 'OD.hpc.cust:frac_W')

        self.prob.model.connect('inlet:ram_recovery', 'OD.inlet.ram_recovery')
        # self.prob.model.connect('splitter:BPR', 'OD.splitter.BPR')
        self.prob.model.connect('duct4:dPqP', 'OD.duct4.dPqP')
        self.prob.model.connect('duct6:dPqP', 'OD.duct6.dPqP')
        self.prob.model.connect('burner:dPqP', 'OD.burner.dPqP')
        self.prob.model.connect('duct11:dPqP', 'OD.duct11.dPqP')
        self.prob.model.connect('duct13:dPqP', 'OD.duct13.dPqP')
        self.prob.model.connect('core_nozz:Cv', 'OD.core_nozz.Cv')
        self.prob.model.connect('duct15:dPqP', 'OD.duct15.dPqP')
        self.prob.model.connect('byp_nozz:Cv', 'OD.byp_nozz.Cv')
        self.prob.model.connect('hp_shaft:HPX', 'OD.hp_shaft.HPX')

        self.prob.model.connect('hpc:cool1:frac_W', 'OD.hpc.cool1:frac_W')
        self.prob.model.connect('hpc:cool1:frac_P', 'OD.hpc.cool1:frac_P')
        self.prob.model.connect('hpc:cool1:frac_work', 'OD.hpc.cool1:frac_work')
        self.prob.model.connect('hpc:cool2:frac_W', 'OD.hpc.cool2:frac_W')
        self.prob.model.connect('hpc:cool2:frac_P', 'OD.hpc.cool2:frac_P')
        self.prob.model.connect('hpc:cool2:frac_work', 'OD.hpc.cool2:frac_work')
        self.prob.model.connect('bld3:cool3:frac_W', 'OD.bld3.cool3:frac_W')
        self.prob.model.connect('bld3:cool4:frac_W', 'OD.bld3.cool4:frac_W')
        # self.prob.model.connect('hpc:cust:frac_W', 'OD.hpc.cust:frac_W')
        self.prob.model.connect('hpc:cust:frac_P', 'OD.hpc.cust:frac_P')
        self.prob.model.connect('hpc:cust:frac_work', 'OD.hpc.cust:frac_work')
        self.prob.model.connect('hpt:cool3:frac_P', 'OD.hpt.cool3:frac_P')
        self.prob.model.connect('hpt:cool4:frac_P', 'OD.hpt.cool4:frac_P')
        self.prob.model.connect('lpt:cool1:frac_P', 'OD.lpt.cool1:frac_P')
        self.prob.model.connect('lpt:cool2:frac_P', 'OD.lpt.cool2:frac_P')
        self.prob.model.connect('bypBld:frac_W', 'OD.byp_bld.bypBld:frac_W')

        self.prob.model.connect('fan:s_PRdes', 'OD.fan.s_PR')
        self.prob.model.connect('fan:s_WcDes', 'OD.fan.s_Wc')
        self.prob.model.connect('fan:s_effDes', 'OD.fan.s_eff')
        self.prob.model.connect('fan:s_NcDes', 'OD.fan.s_Nc')
        self.prob.model.connect('lpc:s_PRdes', 'OD.lpc.s_PR')
        self.prob.model.connect('lpc:s_WcDes', 'OD.lpc.s_Wc')
        self.prob.model.connect('lpc:s_effDes', 'OD.lpc.s_eff')
        self.prob.model.connect('lpc:s_NcDes', 'OD.lpc.s_Nc')
        self.prob.model.connect('hpc:s_PRdes', 'OD.hpc.s_PR')
        self.prob.model.connect('hpc:s_WcDes', 'OD.hpc.s_Wc')
        self.prob.model.connect('hpc:s_effDes', 'OD.hpc.s_eff')
        self.prob.model.connect('hpc:s_NcDes', 'OD.hpc.s_Nc')
        self.prob.model.connect('hpt:s_PRdes', 'OD.hpt.s_PR')
        self.prob.model.connect('hpt:s_WpDes', 'OD.hpt.s_Wp')
        self.prob.model.connect('hpt:s_effDes', 'OD.hpt.s_eff')
        self.prob.model.connect('hpt:s_NpDes', 'OD.hpt.s_Np')
        self.prob.model.connect('lpt:s_PRdes', 'OD.lpt.s_PR')
        self.prob.model.connect('lpt:s_WpDes', 'OD.lpt.s_Wp')
        self.prob.model.connect('lpt:s_effDes', 'OD.lpt.s_eff')
        self.prob.model.connect('lpt:s_NpDes', 'OD.lpt.s_Np')

        self.prob.model.connect('core_nozz:Throat:stat:area', 'OD.balance.rhs:W')
        self.prob.model.connect('byp_nozz:Throat:stat:area', 'OD.balance.rhs:BPR')

        self.prob.model.connect('inlet:Fl_O:stat:area', 'OD.inlet.area')
        self.prob.model.connect('fan:Fl_O:stat:area', 'OD.fan.area')
        self.prob.model.connect('splitter:Fl_O1:stat:area', 'OD.splitter.area1')
        self.prob.model.connect('splitter:Fl_O2:stat:area', 'OD.splitter.area2')
        self.prob.model.connect('duct4:Fl_O:stat:area', 'OD.duct4.area')
        self.prob.model.connect('lpc:Fl_O:stat:area', 'OD.lpc.area')
        self.prob.model.connect('duct6:Fl_O:stat:area', 'OD.duct6.area')
        self.prob.model.connect('hpc:Fl_O:stat:area', 'OD.hpc.area')
        self.prob.model.connect('bld3:Fl_O:stat:area', 'OD.bld3.area')
        self.prob.model.connect('burner:Fl_O:stat:area', 'OD.burner.area')
        self.prob.model.connect('hpt:Fl_O:stat:area', 'OD.hpt.area')
        self.prob.model.connect('duct11:Fl_O:stat:area', 'OD.duct11.area')
        self.prob.model.connect('lpt:Fl_O:stat:area', 'OD.lpt.area')
        self.prob.model.connect('duct13:Fl_O:stat:area', 'OD.duct13.area')
        self.prob.model.connect('byp_bld:Fl_O:stat:area', 'OD.byp_bld.area')
        self.prob.model.connect('duct15:Fl_O:stat:area', 'OD.duct15.area')

        self.prob.set_solver_print(level=-1)
        self.prob.set_solver_print(level=2, depth=1)
        self.prob.setup(check=False)

        # from openmdao.api import view_model
        # view_model(self.prob)
        # exit()

    def benchmark_case1(self):
        # ADP Point
        np.seterr(divide='raise')

        self.prob['OD.balance.FAR'] = 0.02467
        self.prob['OD.balance.W'] = 320.931
        self.prob['OD.balance.BPR'] = 5.105
        self.prob['OD.balance.lp_Nmech'] = 4666.1
        self.prob['OD.balance.hp_Nmech'] = 14705.7
        self.prob['OD.fc.balance.Pt'] = 5.2
        self.prob['OD.fc.balance.Tt'] = 440.0
        self.prob['OD.hpt.PR'] = 3.6200
        self.prob['OD.lpt.PR'] = 4.3645
        self.prob['OD.fan.map.RlineMap'] = 2.0
        self.prob['OD.lpc.map.RlineMap'] = 2.0
        self.prob['OD.hpc.map.RlineMap'] = 2.0
        self.prob.run_model()
        tol = 1e-3

        print()

        reg_data = 321.251
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 30.094
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.02491
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 14705.7
        pyc = self.prob['OD.balance.hp_Nmech'][0]
        print('HPT Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 4666.1
        pyc = self.prob['OD.balance.lp_Nmech'][0]
        print('LPT Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 13274.4
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.63101
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1276.48
        pyc = self.prob['OD.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 5.105
        pyc = self.prob['OD.balance.BPR'][0]
        print('BPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()

    def benchmark_case2(self):
        # TOC Point
        np.seterr(divide='raise')

        self.prob['OD.balance.FAR'] = 0.02467
        self.prob['OD.balance.W'] = 320.931
        self.prob['OD.balance.BPR'] = 5.105
        self.prob['OD.balance.lp_Nmech'] = 4666.1
        self.prob['OD.balance.hp_Nmech'] = 14705.7
        self.prob['OD.fc.balance.Pt'] = 5.2
        self.prob['OD.fc.balance.Tt'] = 440.0
        self.prob['OD.hpt.PR'] = 3.6200
        self.prob['OD.lpt.PR'] = 4.3645
        self.prob['OD.fan.map.RlineMap'] = 2.0
        self.prob['OD.lpc.map.RlineMap'] = 2.0
        self.prob['OD.hpc.map.RlineMap'] = 2.0

        self.prob['OD_MN'] = 0.8
        self.prob['OD_alt'] = 35000.0
        self.prob['OD_Fn_target'] = 5970.0
        self.prob['OD_dTs'] = 0.0
        self.prob['OD_cust_fracW'] = 0.0422
        self.prob.run_model()
        tol = 1e-3

        print()

        reg_data = 327.265
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 32.415
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.02616
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 14952.3
        pyc = self.prob['OD.balance.hp_Nmech'][0]
        print('HPT Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 4933.4
        pyc = self.prob['OD.balance.lp_Nmech'][0]
        print('LPT Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 13889.9
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.64539
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1317.31
        pyc = self.prob['OD.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 4.898
        pyc = self.prob['OD.balance.BPR'][0]
        print('BPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()

    def benchmark_case3(self):
        # RTO Point
        np.seterr(divide='raise')

        self.prob['OD.balance.FAR'] = 0.03165
        self.prob['OD.balance.W'] = 810.83
        self.prob['OD.balance.BPR'] = 5.1053
        self.prob['OD.balance.lp_Nmech'] = 4975.9
        self.prob['OD.balance.hp_Nmech'] = 16230.1
        self.prob['OD.fc.balance.Pt'] = 15.349
        self.prob['OD.fc.balance.Tt'] = 552.49
        self.prob['OD.hpt.PR'] = 3.591
        self.prob['OD.lpt.PR'] = 4.173
        self.prob['OD.fan.map.RlineMap'] = 2.0
        self.prob['OD.lpc.map.RlineMap'] = 2.0
        self.prob['OD.hpc.map.RlineMap'] = 2.0

        self.prob['OD_MN'] = 0.25
        self.prob['OD_alt'] = 0.0
        self.prob['OD_Fn_target'] = 22590.0
        self.prob['OD_dTs'] = 27.0
        self.prob['OD_cust_fracW'] = 0.0177
        self.prob.run_model()
        tol = 1e-3

        print()

        reg_data = 825.049
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 28.998
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.02975
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 16222.1
        pyc = self.prob['OD.balance.hp_Nmech'][0]
        print('HPT Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 5050
        pyc = self.prob['OD.balance.lp_Nmech'][0]
        print('LPT Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 29930.8
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.47488
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1536.94
        pyc = self.prob['OD.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 5.243
        pyc = self.prob['OD.balance.BPR'][0]
        print('BPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()

    def benchmark_case4(self):
        # SLS Point
        np.seterr(divide='raise')

        self.prob['OD.balance.FAR'] = 0.03114
        self.prob['OD.balance.W'] = 771.34
        self.prob['OD.balance.BPR'] = 5.0805
        self.prob['OD.balance.lp_Nmech'] = 4912.7
        self.prob['OD.balance.hp_Nmech'] = 16106.9
        self.prob['OD.fc.balance.Pt'] = 14.696
        self.prob['OD.fc.balance.Tt'] = 545.67
        self.prob['OD.hpt.PR'] = 3.595
        self.prob['OD.lpt.PR'] = 4.147
        self.prob['OD.fan.map.RlineMap'] = 2.0
        self.prob['OD.lpc.map.RlineMap'] = 2.0
        self.prob['OD.hpc.map.RlineMap'] = 2.0

        self.prob['OD_MN'] = 0.00001
        self.prob['OD_alt'] = 0.0
        self.prob['OD_Fn_target'] = 27113.0
        self.prob['OD_dTs'] = 27.0
        self.prob['OD_cust_fracW'] = 0.0185
        self.prob.run_model()
        tol = 1e-3

        print()

        reg_data = 786.741
        pyc = self.prob['OD.inlet.Fl_O:stat:W'][0]
        print('W:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 28.418
        pyc = self.prob['OD.perf.OPR'][0]
        print('OPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.02912
        pyc = self.prob['OD.balance.FAR'][0]
        print('FAR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 16065.1
        pyc = self.prob['OD.balance.hp_Nmech'][0]
        print('HPT Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 4949.1
        pyc = self.prob['OD.balance.lp_Nmech'][0]
        print('LPT Nmech:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 27113.3
        pyc = self.prob['OD.perf.Fg'][0]
        print('Fg:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 0.36662
        pyc = self.prob['OD.perf.TSFC'][0]
        print('TSFC:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 1509.41
        pyc = self.prob['OD.bld3.Fl_O:tot:T'][0]
        print('Tt3:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        reg_data = 5.282
        pyc = self.prob['OD.balance.BPR'][0]
        print('BPR:', reg_data, pyc)
        assert_rel_error(self, pyc, reg_data, tol)

        print()

if __name__ == "__main__":
    unittest.main()
