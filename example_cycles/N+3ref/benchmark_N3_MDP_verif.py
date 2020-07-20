import numpy as np 
import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import pycycle.api as pyc

from N3_MDP_verif import N3_MDP_verif_model

class N3MDPVerifTestCase(unittest.TestCase):

    def benchmark_case1(self):

        OD_statics = True
        prob = N3_MDP_verif_model(OD_statics)
        prob.setup(check=False)

        prob.set_val('TOC.splitter.BPR', 23.7281)
        prob.set_val('TOC.balance.rhs:hpc_PR', 55.0)
        prob.set_val('SLS.fc.MN', 0.001)
        prob.set_val('SLS.balance.rhs:FAR', 28620.9, units='lbf')
        prob.set_val('CRZ.balance.rhs:FAR', 5466.5, units='lbf')
        prob.set_val('bal.rhs:TOC_BPR', 1.40)
        prob.set_val('fan:PRdes', 1.300)
        prob.set_val('TOC.opr_calc.FPR', 1.300)
        prob.set_val('lpc:PRdes', 3.000),
        prob.set_val('TOC.opr_calc.LPCPR', 3.000)
        prob.set_val('T4_ratio.TR', 0.926470588)

        prob['RTO.hpt_cooling.x_factor'] = 0.9

        # initial guesses
        prob['TOC.balance.FAR'] = 0.02650
        prob['bal.TOC_W'] = 820.95
        prob['TOC.balance.lpt_PR'] = 10.937
        prob['TOC.balance.hpt_PR'] = 4.185
        prob['TOC.fc.balance.Pt'] = 5.272
        prob['TOC.fc.balance.Tt'] = 444.41

        prob['RTO.balance.FAR'] = 0.02832
        prob['RTO.balance.W'] = 1916.13
        prob['RTO.balance.BPR'] = 25.5620
        prob['RTO.balance.fan_Nmech'] = 2132.6
        prob['RTO.balance.lp_Nmech'] = 6611.2
        prob['RTO.balance.hp_Nmech'] = 22288.2
        prob['RTO.fc.balance.Pt'] = 15.349
        prob['RTO.fc.balance.Tt'] = 552.49
        prob['RTO.hpt.PR'] = 4.210
        prob['RTO.lpt.PR'] = 8.161
        prob['RTO.fan.map.RlineMap'] = 1.7500
        prob['RTO.lpc.map.RlineMap'] = 2.0052
        prob['RTO.hpc.map.RlineMap'] = 2.0589
        prob['RTO.gearbox.trq_base'] = 52509.1  

        prob['SLS.balance.FAR'] = 0.02541
        prob['SLS.balance.W'] = 1734.44
        prob['SLS.balance.BPR'] = 27.3467
        prob['SLS.balance.fan_Nmech'] = 1953.1
        prob['SLS.balance.lp_Nmech'] = 6054.5
        prob['SLS.balance.hp_Nmech'] = 21594.0
        prob['SLS.fc.balance.Pt'] = 14.696
        prob['SLS.fc.balance.Tt'] = 545.67
        prob['SLS.hpt.PR'] = 4.245
        prob['SLS.lpt.PR'] = 7.001
        prob['SLS.fan.map.RlineMap'] = 1.7500
        prob['SLS.lpc.map.RlineMap'] = 1.8632
        prob['SLS.hpc.map.RlineMap'] = 2.0281   
        prob['SLS.gearbox.trq_base'] = 41779.4

        prob['CRZ.balance.FAR'] = 0.02510
        prob['CRZ.balance.W'] = 802.79
        prob['CRZ.balance.BPR'] = 24.3233
        prob['CRZ.balance.fan_Nmech'] = 2118.7
        prob['CRZ.balance.lp_Nmech'] = 6567.9
        prob['CRZ.balance.hp_Nmech'] = 20574.1
        prob['CRZ.fc.balance.Pt'] = 5.272
        prob['CRZ.fc.balance.Tt'] = 444.41
        prob['CRZ.hpt.PR'] = 4.197
        prob['CRZ.lpt.PR'] = 10.803
        prob['CRZ.fan.map.RlineMap'] = 1.9397
        prob['CRZ.lpc.map.RlineMap'] = 2.1075   
        prob['CRZ.hpc.map.RlineMap'] = 1.9746
        prob['CRZ.gearbox.trq_base'] = 22369.7

        prob.run_model()

        tol = 1e-4
    
        assert_near_equal(prob['TOC.inlet.Fl_O:stat:W'], 820.92037027, tol)#
        assert_near_equal(prob['TOC.inlet.Fl_O:tot:P'], 5.26210728, tol)#
        assert_near_equal(prob['TOC.hpc.Fl_O:tot:P'], 282.22391512, tol)#
        assert_near_equal(prob['TOC.burner.Wfuel'], 0.74642969, tol)#
        assert_near_equal(prob['TOC.inlet.F_ram'], 19866.48480269, tol)#
        assert_near_equal(prob['TOC.core_nozz.Fg'], 1556.44941177, tol)#
        assert_near_equal(prob['TOC.byp_nozz.Fg'], 24436.69962673, tol)#
        assert_near_equal(prob['TOC.perf.TSFC'], 0.43859869, tol)#
        assert_near_equal(prob['TOC.perf.OPR'], 53.63325, tol)#
        assert_near_equal(prob['TOC.balance.FAR'], 0.02650755, tol)#
        assert_near_equal(prob['TOC.hpc.Fl_O:tot:T'], 1530.58386828, tol)#
        assert_near_equal(prob['RTO.inlet.Fl_O:stat:W'], 1916.01614631, tol)#
        assert_near_equal(prob['RTO.inlet.Fl_O:tot:P'], 15.3028198, tol)#
        assert_near_equal(prob['RTO.hpc.Fl_O:tot:P'], 638.95720683, tol)#
        assert_near_equal(prob['RTO.burner.Wfuel'], 1.73329552, tol)#
        assert_near_equal(prob['RTO.inlet.F_ram'], 17047.53270726, tol)#
        assert_near_equal(prob['RTO.core_nozz.Fg'], 2220.78852731, tol)#
        assert_near_equal(prob['RTO.byp_nozz.Fg'], 37626.74417995, tol)#
        assert_near_equal(prob['RTO.perf.TSFC'], 0.27367824, tol)#
        assert_near_equal(prob['RTO.perf.OPR'], 41.7542136, tol)#
        assert_near_equal(prob['RTO.balance.FAR'], 0.02832782, tol)#
        assert_near_equal(prob['RTO.balance.fan_Nmech'], 2132.71615737, tol)#
        assert_near_equal(prob['RTO.balance.lp_Nmech'], 6611.46890258, tol)#
        assert_near_equal(prob['RTO.balance.hp_Nmech'], 22288.52228766, tol)#
        assert_near_equal(prob['RTO.hpc.Fl_O:tot:T'], 1721.14599533, tol)#
        assert_near_equal(prob['SLS.inlet.Fl_O:stat:W'], 1735.5273, tol)#
        assert_near_equal(prob['SLS.inlet.Fl_O:tot:P'], 14.62243072, tol)#
        assert_near_equal(prob['SLS.hpc.Fl_O:tot:P'], 522.99027178, tol)#
        assert_near_equal(prob['SLS.burner.Wfuel'], 1.32250739, tol)#
        assert_near_equal(prob['SLS.inlet.F_ram'], 61.76661874, tol)#
        assert_near_equal(prob['SLS.core_nozz.Fg'], 1539.99663, tol)#
        assert_near_equal(prob['SLS.byp_nozz.Fg'], 27142.60997896, tol)#
        assert_near_equal(prob['SLS.perf.TSFC'], 0.16634825, tol)#
        assert_near_equal(prob['SLS.perf.OPR'], 35.76630191, tol)#
        assert_near_equal(prob['SLS.balance.FAR'], 0.02544378, tol)#
        assert_near_equal(prob['SLS.balance.fan_Nmech'], 1954.97855672, tol)#
        assert_near_equal(prob['SLS.balance.lp_Nmech'], 6060.47827242, tol)#
        assert_near_equal(prob['SLS.balance.hp_Nmech'], 21601.07508077, tol)#
        assert_near_equal(prob['SLS.hpc.Fl_O:tot:T'], 1628.85845903, tol)#
        assert_near_equal(prob['CRZ.inlet.Fl_O:stat:W'], 802.76200548, tol)#
        assert_near_equal(prob['CRZ.inlet.Fl_O:tot:P'], 5.26210728, tol)#
        assert_near_equal(prob['CRZ.hpc.Fl_O:tot:P'], 264.63649163, tol)#
        assert_near_equal(prob['CRZ.burner.Wfuel'], 0.67514702, tol)#
        assert_near_equal(prob['CRZ.inlet.F_ram'], 19427.04768877, tol)#
        assert_near_equal(prob['CRZ.core_nozz.Fg'], 1383.93102366, tol)#
        assert_near_equal(prob['CRZ.byp_nozz.Fg'], 23557.11447723, tol)#
        assert_near_equal(prob['CRZ.perf.TSFC'], 0.44079257, tol)#
        assert_near_equal(prob['CRZ.perf.OPR'], 50.29097236, tol)#
        assert_near_equal(prob['CRZ.balance.FAR'], 0.02510864, tol)#    
        assert_near_equal(prob['CRZ.balance.fan_Nmech'], 2118.65676797, tol)#
        assert_near_equal(prob['CRZ.balance.lp_Nmech'], 6567.88447364, tol)#
        assert_near_equal(prob['CRZ.balance.hp_Nmech'], 20574.08438737, tol)#
        assert_near_equal(prob['CRZ.hpc.Fl_O:tot:T'], 1494.29261337, tol)#


if __name__ == "__main__":
    unittest.main()