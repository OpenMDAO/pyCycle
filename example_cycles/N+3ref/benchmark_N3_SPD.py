import numpy as np 
import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import pycycle.api as pyc

from N3_SPD import N3_SPD_model

class N3MDPOptTestCase(unittest.TestCase):

    def benchmark_case1(self):

        prob = N3_SPD_model()

        prob.setup()

        # Define the design point
        prob.set_val('TOC.splitter.BPR', 23.94514401),
        prob.set_val('TOC.balance.rhs:hpc_PR', 53.6332)
        prob.set_val('TOC.fc.W', 820.44097898, units='lbm/s') 

        # Set specific cycle parameters
        prob.set_val('fan:PRdes', 1.300),
        prob.set_val('lpc:PRdes', 3.000),
        prob.set_val('T4_ratio.TR', 0.926470588)
        prob.set_val('RTO_T4', 3400.0, units='degR')
        prob.set_val('SLS.balance.rhs:FAR', 28620.84, units='lbf')
        prob.set_val('CRZ.balance.rhs:FAR', 5510.72833567, units='lbf') 
        prob.set_val('RTO.hpt_cooling.x_factor', 0.9)

        # Set initial guesses for balances
        prob['TOC.balance.FAR'] = 0.02650
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

        tol = 5e-4

        assert_near_equal(prob['TOC.inlet.Fl_O:tot:P'], 5.26210728, tol)#
        assert_near_equal(prob['TOC.hpc.Fl_O:tot:P'], 275.21039426, tol)#
        assert_near_equal(prob['TOC.burner.Wfuel'], 0.74668441, tol)#
        assert_near_equal(prob['TOC.inlet.F_ram'], 19854.88340967, tol)#
        assert_near_equal(prob['TOC.core_nozz.Fg'], 1547.13847348, tol)#
        assert_near_equal(prob['TOC.byp_nozz.Fg'], 24430.7872167, tol)#
        assert_near_equal(prob['TOC.perf.TSFC'], 0.43900789, tol)#
        assert_near_equal(prob['TOC.perf.OPR'], 52.30041498, tol)#
        assert_near_equal(prob['TOC.balance.FAR'], 0.02669913, tol)#
        assert_near_equal(prob['TOC.hpc.Fl_O:tot:T'], 1517.98001185, tol)#
        assert_near_equal(prob['RTO.inlet.Fl_O:stat:W'], 1915.22687254, tol)#
        assert_near_equal(prob['RTO.inlet.Fl_O:tot:P'], 15.3028198, tol)#
        assert_near_equal(prob['RTO.hpc.Fl_O:tot:P'], 623.4047562, tol)#
        assert_near_equal(prob['RTO.burner.Wfuel'], 1.73500397, tol)#
        assert_near_equal(prob['RTO.inlet.F_ram'], 17040.51023484, tol)#
        assert_near_equal(prob['RTO.core_nozz.Fg'], 2208.78618847, tol)#
        assert_near_equal(prob['RTO.byp_nozz.Fg'], 37631.72404627, tol)#
        assert_near_equal(prob['RTO.perf.TSFC'], 0.273948, tol)#
        assert_near_equal(prob['RTO.perf.OPR'], 40.73790087, tol)#
        assert_near_equal(prob['RTO.balance.FAR'], 0.02853677, tol)#
        assert_near_equal(prob['RTO.balance.fan_Nmech'], 2133.2082055, tol)#
        assert_near_equal(prob['RTO.balance.lp_Nmech'], 6612.99426306, tol)#
        assert_near_equal(prob['RTO.balance.hp_Nmech'], 22294.4214065, tol)#
        assert_near_equal(prob['RTO.hpc.Fl_O:tot:T'], 1707.8424979, tol)#
        assert_near_equal(prob['SLS.inlet.Fl_O:stat:W'], 1733.66974646, tol)#
        assert_near_equal(prob['SLS.inlet.Fl_O:tot:P'], 14.62242048, tol)#
        assert_near_equal(prob['SLS.hpc.Fl_O:tot:P'], 509.33522171, tol)#
        assert_near_equal(prob['SLS.burner.Wfuel'], 1.3201035, tol)#
        assert_near_equal(prob['SLS.inlet.F_ram'], 0.06170051, tol)#
        assert_near_equal(prob['SLS.core_nozz.Fg'], 1526.45701492, tol)#
        assert_near_equal(prob['SLS.byp_nozz.Fg'], 27094.44468547, tol)#
        assert_near_equal(prob['SLS.perf.TSFC'], 0.16604588, tol)#
        assert_near_equal(prob['SLS.perf.OPR'], 34.8324836, tol)#
        assert_near_equal(prob['SLS.balance.FAR'], 0.02559136, tol)#
        assert_near_equal(prob['SLS.balance.fan_Nmech'], 1953.67749381, tol)#
        assert_near_equal(prob['SLS.balance.lp_Nmech'], 6056.44494761, tol)#
        assert_near_equal(prob['SLS.balance.hp_Nmech'], 21599.4239832, tol)#
        assert_near_equal(prob['SLS.hpc.Fl_O:tot:T'], 1615.20710655, tol)#
        assert_near_equal(prob['CRZ.inlet.Fl_O:stat:W'], 802.28669491, tol)#
        assert_near_equal(prob['CRZ.inlet.Fl_O:tot:P'], 5.26210728, tol)#
        assert_near_equal(prob['CRZ.hpc.Fl_O:tot:P'], 258.04394098, tol)#
        assert_near_equal(prob['CRZ.burner.Wfuel'], 0.67533764, tol)#
        assert_near_equal(prob['CRZ.inlet.F_ram'], 19415.54505032, tol)#
        assert_near_equal(prob['CRZ.core_nozz.Fg'], 1375.4427785, tol)#
        assert_near_equal(prob['CRZ.byp_nozz.Fg'], 23550.83060746, tol)#
        assert_near_equal(prob['CRZ.perf.TSFC'], 0.44117862, tol)#
        assert_near_equal(prob['CRZ.perf.OPR'], 49.03813765, tol)#
        assert_near_equal(prob['CRZ.balance.FAR'], 0.02528878, tol)#
        assert_near_equal(prob['CRZ.balance.fan_Nmech'], 2118.62554023, tol)#
        assert_near_equal(prob['CRZ.balance.lp_Nmech'], 6567.78766693, tol)#
        assert_near_equal(prob['CRZ.balance.hp_Nmech'], 20574.43651756, tol)#
        assert_near_equal(prob['CRZ.hpc.Fl_O:tot:T'], 1481.9756247, tol)#   


if __name__ == "__main__":
    unittest.main()