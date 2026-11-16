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

        prob.setup()

        # Define the design point
        prob.set_val('TOC.splitter.BPR', 23.7281)
        prob.set_val('TOC.balance.rhs:hpc_PR', 53.6332)

        # Set specific cycle parameters
        prob.set_val('fan:PRdes', 1.300)
        prob.set_val('SLS.balance.rhs:FAR', 28620.9, units='lbf') 
        prob.set_val('CRZ.balance.rhs:FAR', 5466.5, units='lbf')
        prob.set_val('lpc:PRdes', 3.000),
        prob.set_val('T4_ratio.TR', 0.926470588)
        prob.set_val('bal.mult:TOC_BPR', 1.41038)
        prob.set_val('RTO.hpt_cooling.x_factor', 0.9)

        # Set initial guesses for balances
        prob['TOC.balance.FAR'] = 0.02650
        prob['bal.TOC_W'] = 820.95
        prob['TOC.balance.lpt_PR'] = 10.937
        prob['TOC.balance.hpt_PR'] = 4.185
        prob['TOC.fc.balance.Pt'] = 5.272
        prob['TOC.fc.balance.Tt'] = 444.41

        FAR_guess = [0.02832, 0.02541, 0.02510]
        W_guess = [1916.13, 2000. , 802.79]
        BPR_guess = [25.5620, 27.3467, 24.3233]
        fan_Nmech_guess = [2132.6, 1953.1, 2118.7]
        lp_Nmech_guess = [6611.2, 6054.5, 6567.9]
        hp_Nmech_guess = [22288.2, 21594.0, 20574.1]
        Pt_guess = [15.349, 14.696, 5.272]
        Tt_guess = [552.49, 545.67, 444.41]
        hpt_PR_guess = [4.210, 4.245, 4.197]
        lpt_PR_guess = [8.161, 7.001, 10.803]
        fan_Rline_guess = [1.7500, 1.7500, 1.9397]
        lpc_Rline_guess = [2.0052, 1.8632, 2.1075]
        hpc_Rline_guess = [2.0589, 2.0281, 1.9746]
        trq_guess = [52509.1, 41779.4, 22369.7]

        for i, pt in enumerate(prob.model.od_pts):

            # initial guesses
            prob[pt+'.balance.FAR'] = FAR_guess[i]
            prob[pt+'.balance.W'] = W_guess[i]
            prob[pt+'.balance.BPR'] = BPR_guess[i]
            prob[pt+'.balance.fan_Nmech'] = fan_Nmech_guess[i]
            prob[pt+'.balance.lp_Nmech'] = lp_Nmech_guess[i]
            prob[pt+'.balance.hp_Nmech'] = hp_Nmech_guess[i]
            prob[pt+'.fc.balance.Pt'] = Pt_guess[i]
            prob[pt+'.fc.balance.Tt'] = Tt_guess[i]
            prob[pt+'.hpt.PR'] = hpt_PR_guess[i]
            prob[pt+'.lpt.PR'] = lpt_PR_guess[i]
            prob[pt+'.fan.map.RlineMap'] = fan_Rline_guess[i]
            prob[pt+'.lpc.map.RlineMap'] = lpc_Rline_guess[i]
            prob[pt+'.hpc.map.RlineMap'] = hpc_Rline_guess[i]
            prob[pt+'.gearbox.trq_base'] = trq_guess[i]

        prob.run_model()

        tol = 1e-4
        
        assert_near_equal(prob['TOC.inlet.Fl_O:stat:W'], 820.48630223, tol)#
        assert_near_equal(prob['TOC.inlet.Fl_O:tot:P'], 5.26210728, tol)#
        assert_near_equal(prob['TOC.hpc.Fl_O:tot:P'], 275.21039426, tol)#
        assert_near_equal(prob['TOC.burner.Wfuel'], 0.74737594, tol)#
        assert_near_equal(prob['TOC.inlet.F_ram'], 19855.97955497, tol)#
        assert_near_equal(prob['TOC.core_nozz.Fg'], 1549.40273807, tol)#
        assert_near_equal(prob['TOC.byp_nozz.Fg'], 24430.65940258, tol)#
        assert_near_equal(prob['TOC.perf.TSFC'], 0.43933983, tol)#
        assert_near_equal(prob['TOC.perf.OPR'], 52.30041498, tol)#
        assert_near_equal(prob['TOC.balance.FAR'], 0.02671373, tol)#
        assert_near_equal(prob['TOC.hpc.Fl_O:tot:T'], 1517.81289807, tol)#

        assert_near_equal(prob['RTO.inlet.Fl_O:stat:W'], 1915.27525687, tol)#
        assert_near_equal(prob['RTO.inlet.Fl_O:tot:P'], 15.3028198, tol)#
        assert_near_equal(prob['RTO.hpc.Fl_O:tot:P'], 623.36975929, tol)#
        assert_near_equal(prob['RTO.burner.Wfuel'], 1.73655658, tol)#
        assert_near_equal(prob['RTO.inlet.F_ram'], 17040.94013754, tol)#
        assert_near_equal(prob['RTO.core_nozz.Fg'], 2211.9022128, tol)#
        assert_near_equal(prob['RTO.byp_nozz.Fg'], 37629.03792478, tol)#
        assert_near_equal(prob['RTO.perf.TSFC'], 0.27419314, tol)#
        assert_near_equal(prob['RTO.perf.OPR'], 40.73561391, tol)#
        assert_near_equal(prob['RTO.balance.FAR'], 0.02855312, tol)#
        assert_near_equal(prob['RTO.balance.fan_Nmech'], 2133.12249824, tol)#
        assert_near_equal(prob['RTO.balance.lp_Nmech'], 6612.72856858, tol)#
        assert_near_equal(prob['RTO.balance.hp_Nmech'], 22294.06683282, tol)#
        assert_near_equal(prob['RTO.hpc.Fl_O:tot:T'], 1707.62243199, tol)#

        assert_near_equal(prob['SLS.inlet.Fl_O:stat:W'], 1733.71717503, tol)#
        assert_near_equal(prob['SLS.inlet.Fl_O:tot:P'], 14.62242048, tol)#
        assert_near_equal(prob['SLS.hpc.Fl_O:tot:P'], 509.31503900, tol)#
        assert_near_equal(prob['SLS.burner.Wfuel'], 1.32137091, tol)#
        assert_near_equal(prob['SLS.inlet.F_ram'], 0.06170219, tol)#
        assert_near_equal(prob['SLS.core_nozz.Fg'], 1528.71136438, tol)#
        assert_near_equal(prob['SLS.byp_nozz.Fg'], 27092.19033781, tol)#
        assert_near_equal(prob['SLS.perf.TSFC'], 0.16620530, tol)#
        assert_near_equal(prob['SLS.perf.OPR'], 34.8311033, tol)#
        assert_near_equal(prob['SLS.balance.FAR'], 0.02560763, tol)#
        assert_near_equal(prob['SLS.balance.fan_Nmech'], 1953.60290521, tol)#
        assert_near_equal(prob['SLS.balance.lp_Nmech'], 6056.21372126, tol)#
        assert_near_equal(prob['SLS.balance.hp_Nmech'], 21599.20227635, tol)#
        assert_near_equal(prob['SLS.hpc.Fl_O:tot:T'], 1615.01599167, tol)#

        assert_near_equal(prob['CRZ.inlet.Fl_O:stat:W'], 802.33271677, tol)#
        assert_near_equal(prob['CRZ.inlet.Fl_O:tot:P'], 5.26210728, tol)#
        assert_near_equal(prob['CRZ.hpc.Fl_O:tot:P'], 258.04606511, tol)#
        assert_near_equal(prob['CRZ.burner.Wfuel'], 0.67597183, tol)#
        assert_near_equal(prob['CRZ.inlet.F_ram'], 19416.65811754, tol)#
        assert_near_equal(prob['CRZ.core_nozz.Fg'], 1377.49290443, tol)#
        assert_near_equal(prob['CRZ.byp_nozz.Fg'], 23550.83954023, tol)#
        assert_near_equal(prob['CRZ.perf.TSFC'], 0.44151713, tol)#
        assert_near_equal(prob['CRZ.perf.OPR'], 49.03854131, tol)#
        assert_near_equal(prob['CRZ.balance.FAR'], 0.02530275, tol)#
        assert_near_equal(prob['CRZ.balance.fan_Nmech'], 2118.63368521, tol)#
        assert_near_equal(prob['CRZ.balance.lp_Nmech'], 6567.81291657, tol)#
        assert_near_equal(prob['CRZ.balance.hp_Nmech'], 20574.44969253, tol)#
        assert_near_equal(prob['CRZ.hpc.Fl_O:tot:T'], 1481.81777838, tol)#



if __name__ == "__main__":
    unittest.main()