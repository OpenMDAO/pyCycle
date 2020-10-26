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
        
        assert_near_equal(prob['TOC.inlet.Fl_O:stat:W'], 820.44097898, tol)#
        assert_near_equal(prob['TOC.inlet.Fl_O:tot:P'], 5.26210728, tol)#
        assert_near_equal(prob['TOC.hpc.Fl_O:tot:P'], 275.21039426, tol)#
        assert_near_equal(prob['TOC.burner.Wfuel'], 0.74702034, tol)#
        assert_near_equal(prob['TOC.inlet.F_ram'], 19854.83873204, tol)#
        assert_near_equal(prob['TOC.core_nozz.Fg'], 1547.14500321, tol)#
        assert_near_equal(prob['TOC.byp_nozz.Fg'], 24430.78721659, tol)#
        assert_near_equal(prob['TOC.perf.TSFC'], 0.43920593, tol)#
        assert_near_equal(prob['TOC.perf.OPR'], 52.30041498, tol)#
        assert_near_equal(prob['TOC.balance.FAR'], 0.02671119, tol)#
        assert_near_equal(prob['TOC.hpc.Fl_O:tot:T'], 1517.97985269, tol)#

        assert_near_equal(prob['RTO.inlet.Fl_O:stat:W'], 1915.22359344, tol)#
        assert_near_equal(prob['RTO.inlet.Fl_O:tot:P'], 15.3028198, tol)#
        assert_near_equal(prob['RTO.hpc.Fl_O:tot:P'], 623.40703024, tol)#
        assert_near_equal(prob['RTO.burner.Wfuel'], 1.73578775, tol)#
        assert_near_equal(prob['RTO.inlet.F_ram'], 17040.48046811, tol)#
        assert_near_equal(prob['RTO.core_nozz.Fg'], 2208.8023950, tol)#
        assert_near_equal(prob['RTO.byp_nozz.Fg'], 37631.67807307, tol)#
        assert_near_equal(prob['RTO.perf.TSFC'], 0.27407175, tol)#
        assert_near_equal(prob['RTO.perf.OPR'], 40.73804947, tol)#
        assert_near_equal(prob['RTO.balance.FAR'], 0.02854964, tol)#
        assert_near_equal(prob['RTO.balance.fan_Nmech'], 2133.20964469, tol)#
        assert_near_equal(prob['RTO.balance.lp_Nmech'], 6612.99872459, tol)#
        assert_near_equal(prob['RTO.balance.hp_Nmech'], 22294.43280596, tol)#
        assert_near_equal(prob['RTO.hpc.Fl_O:tot:T'], 1707.84433893, tol)#

        assert_near_equal(prob['SLS.inlet.Fl_O:stat:W'], 1733.66701727, tol)#
        assert_near_equal(prob['SLS.inlet.Fl_O:tot:P'], 14.62242048, tol)#
        assert_near_equal(prob['SLS.hpc.Fl_O:tot:P'], 509.33689017, tol)#
        assert_near_equal(prob['SLS.burner.Wfuel'], 1.32070102, tol)#
        assert_near_equal(prob['SLS.inlet.F_ram'], 0.06170041, tol)#
        assert_near_equal(prob['SLS.core_nozz.Fg'], 1526.46929726, tol)#
        assert_near_equal(prob['SLS.byp_nozz.Fg'], 27094.43240315, tol)#
        assert_near_equal(prob['SLS.perf.TSFC'], 0.16612104, tol)#
        assert_near_equal(prob['SLS.perf.OPR'], 34.8325977, tol)#
        assert_near_equal(prob['SLS.balance.FAR'], 0.02560289, tol)#
        assert_near_equal(prob['SLS.balance.fan_Nmech'], 1953.67920923, tol)#
        assert_near_equal(prob['SLS.balance.lp_Nmech'], 6056.45026545, tol)#
        assert_near_equal(prob['SLS.balance.hp_Nmech'], 21599.43696168, tol)#
        assert_near_equal(prob['SLS.hpc.Fl_O:tot:T'], 1615.20862445, tol)#

        assert_near_equal(prob['CRZ.inlet.Fl_O:stat:W'], 802.28514996, tol)#
        assert_near_equal(prob['CRZ.inlet.Fl_O:tot:P'], 5.26210728, tol)#
        assert_near_equal(prob['CRZ.hpc.Fl_O:tot:P'], 258.04448231, tol)#
        assert_near_equal(prob['CRZ.burner.Wfuel'], 0.67564259, tol)#
        assert_near_equal(prob['CRZ.inlet.F_ram'], 19415.50698852, tol)#
        assert_near_equal(prob['CRZ.core_nozz.Fg'], 1375.45106569, tol)#
        assert_near_equal(prob['CRZ.byp_nozz.Fg'], 23550.78724671, tol)#
        assert_near_equal(prob['CRZ.perf.TSFC'], 0.44137759, tol)#
        assert_near_equal(prob['CRZ.perf.OPR'], 49.03824052, tol)#
        assert_near_equal(prob['CRZ.balance.FAR'], 0.02530018, tol)#
        assert_near_equal(prob['CRZ.balance.fan_Nmech'], 2118.62665338, tol)#
        assert_near_equal(prob['CRZ.balance.lp_Nmech'], 6567.79111774, tol)#
        assert_near_equal(prob['CRZ.balance.hp_Nmech'], 20574.44969253, tol)#
        assert_near_equal(prob['CRZ.hpc.Fl_O:tot:T'], 1481.97697491, tol)#



if __name__ == "__main__":
    unittest.main()