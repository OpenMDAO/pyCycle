import numpy as np 
import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import pycycle.api as pyc

from N3_MDP import N3_MDP_model

class N3MDPTestCase(unittest.TestCase):

    def benchmark_case1(self):

        prob = N3_MDP_model()

        prob.setup(check=False)

        prob['RTO.hpt_cooling.x_factor'] = 0.9
        prob.set_val('TOC.splitter.BPR', 23.7281)
        prob.set_val('fan:PRdes', 1.300)
        prob.set_val('SLS.balance.rhs:FAR', 28620.9, units='lbf') 
        prob.set_val('CRZ.balance.rhs:FAR', 5466.5, units='lbf')
        prob.set_val('lpc:PRdes', 3.000),
        prob.set_val('TOC.balance.rhs:hpc_PR', 53.6332)
        prob.set_val('T4_ratio.TR', 0.926470588)
        prob.set_val('bal.mult:TOC_BPR', 1.41038)

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
        prob['SLS.balance.W'] = 2000. #1734.44
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
        
        assert_near_equal(prob['TOC.inlet.Fl_O:stat:W'], 820.44097898, tol)#
        assert_near_equal(prob['TOC.inlet.Fl_O:tot:P'], 5.26210728, tol)#
        assert_near_equal(prob['TOC.hpc.Fl_O:tot:P'], 275.21039426, tol)#
        assert_near_equal(prob['TOC.burner.Wfuel'], 0.74668298, tol)#
        assert_near_equal(prob['TOC.inlet.F_ram'], 19854.88340973, tol)#
        assert_near_equal(prob['TOC.core_nozz.Fg'], 1547.12767722, tol)#
        assert_near_equal(prob['TOC.byp_nozz.Fg'], 24430.78721659, tol)#
        assert_near_equal(prob['TOC.perf.TSFC'], 0.43900782, tol)#
        assert_near_equal(prob['TOC.perf.OPR'], 52.30041498, tol)#
        assert_near_equal(prob['TOC.balance.FAR'], 0.02669913, tol)#
        assert_near_equal(prob['TOC.hpc.Fl_O:tot:T'], 1517.98001183, tol)#

        assert_near_equal(prob['RTO.inlet.Fl_O:stat:W'], 1915.22721047, tol)#
        assert_near_equal(prob['RTO.inlet.Fl_O:tot:P'], 15.3028198, tol)#
        assert_near_equal(prob['RTO.hpc.Fl_O:tot:P'], 623.40626772, tol)#
        assert_near_equal(prob['RTO.burner.Wfuel'], 1.73500332, tol)#
        assert_near_equal(prob['RTO.inlet.F_ram'], 17040.5132416, tol)#
        assert_near_equal(prob['RTO.core_nozz.Fg'], 2208.77420493, tol)#
        assert_near_equal(prob['RTO.byp_nozz.Fg'], 37631.73903667, tol)#
        assert_near_equal(prob['RTO.perf.TSFC'], 0.27394789, tol)#
        assert_near_equal(prob['RTO.perf.OPR'], 40.73799964, tol)#
        assert_near_equal(prob['RTO.balance.FAR'], 0.02853675, tol)#
        assert_near_equal(prob['RTO.balance.fan_Nmech'], 2133.20870996, tol)#
        assert_near_equal(prob['RTO.balance.lp_Nmech'], 6612.99582689, tol)#
        assert_near_equal(prob['RTO.balance.hp_Nmech'], 22294.43364859, tol)#
        assert_near_equal(prob['RTO.hpc.Fl_O:tot:T'], 1707.84422491, tol)#

        assert_near_equal(prob['SLS.inlet.Fl_O:stat:W'], 1733.66992864, tol)#
        assert_near_equal(prob['SLS.inlet.Fl_O:tot:P'], 14.62242048, tol)#
        assert_near_equal(prob['SLS.hpc.Fl_O:tot:P'], 509.33651837, tol)#
        assert_near_equal(prob['SLS.burner.Wfuel'], 1.32010279, tol)#
        assert_near_equal(prob['SLS.inlet.F_ram'], 0.06170051, tol)#
        assert_near_equal(prob['SLS.core_nozz.Fg'], 1526.44902158, tol)#
        assert_near_equal(prob['SLS.byp_nozz.Fg'], 27094.45267893, tol)#
        assert_near_equal(prob['SLS.perf.TSFC'], 0.16604579, tol)#
        assert_near_equal(prob['SLS.perf.OPR'], 34.83257228, tol)#
        assert_near_equal(prob['SLS.balance.FAR'], 0.02559132, tol)#
        assert_near_equal(prob['SLS.balance.fan_Nmech'], 1953.6777749, tol)#
        assert_near_equal(prob['SLS.balance.lp_Nmech'], 6056.44581902, tol)#
        assert_near_equal(prob['SLS.balance.hp_Nmech'], 21599.43268289, tol)#
        assert_near_equal(prob['SLS.hpc.Fl_O:tot:T'], 1615.20840937, tol)#

        assert_near_equal(prob['CRZ.inlet.Fl_O:stat:W'], 802.28690625, tol)#
        assert_near_equal(prob['CRZ.inlet.Fl_O:tot:P'], 5.26210728, tol)#
        assert_near_equal(prob['CRZ.hpc.Fl_O:tot:P'], 258.04461388, tol)#
        assert_near_equal(prob['CRZ.burner.Wfuel'], 0.67533742, tol)#
        assert_near_equal(prob['CRZ.inlet.F_ram'], 19415.55016481, tol)#
        assert_near_equal(prob['CRZ.core_nozz.Fg'], 1375.43593888, tol)#
        assert_near_equal(prob['CRZ.byp_nozz.Fg'], 23550.8425616, tol)#
        assert_near_equal(prob['CRZ.perf.TSFC'], 0.44117847, tol)#
        assert_near_equal(prob['CRZ.perf.OPR'], 49.03826553, tol)#
        assert_near_equal(prob['CRZ.balance.FAR'], 0.02528875, tol)#
        assert_near_equal(prob['CRZ.balance.fan_Nmech'], 2118.62655194, tol)#
        assert_near_equal(prob['CRZ.balance.lp_Nmech'], 6567.79080327, tol)#
        assert_near_equal(prob['CRZ.balance.hp_Nmech'], 20574.44805568, tol)#
        assert_near_equal(prob['CRZ.hpc.Fl_O:tot:T'], 1481.97721995, tol)#


if __name__ == "__main__":
    unittest.main()