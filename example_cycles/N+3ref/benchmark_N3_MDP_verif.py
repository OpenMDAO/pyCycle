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

        prob.setup()

        # Define the design point
        prob.set_val('TOC.splitter.BPR', 23.7281)
        prob.set_val('TOC.balance.rhs:hpc_PR', 55.0)
        prob.set_val('TOC.opr_calc.FPR', 1.300)
        prob.set_val('TOC.opr_calc.LPCPR', 3.000)

        # Set up specific cylce parameters
        prob.set_val('SLS.fc.MN', 0.001)
        prob.set_val('SLS.balance.rhs:FAR', 28620.9, units='lbf')
        prob.set_val('CRZ.balance.rhs:FAR', 5466.5, units='lbf')
        prob.set_val('bal.rhs:TOC_BPR', 1.40)
        prob.set_val('fan:PRdes', 1.300)
        prob.set_val('lpc:PRdes', 3.000),
        prob.set_val('T4_ratio.TR', 0.926470588)

        prob.set_val('RTO.hpt_cooling.x_factor', 0.9)

        # initial guesses
        prob['TOC.balance.FAR'] = 0.02650
        prob['bal.TOC_W'] = 820.95
        prob['TOC.balance.lpt_PR'] = 10.937
        prob['TOC.balance.hpt_PR'] = 4.185
        prob['TOC.fc.balance.Pt'] = 5.272
        prob['TOC.fc.balance.Tt'] = 444.41

        FAR_guess = [0.02832, 0.02541, 0.02510]
        W_guess = [1916.13, 1734.44, 802.79]
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
        hpc_Rline_guess = [2.0589, 2.0281 , 1.9746]
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

        assert_near_equal(prob['TOC.inlet.Fl_O:stat:W'], 820.97022009, tol)#
        assert_near_equal(prob['TOC.inlet.Fl_O:tot:P'], 5.26210728, tol)#
        assert_near_equal(prob['TOC.hpc.Fl_O:tot:P'], 282.22391512, tol)#
        assert_near_equal(prob['TOC.burner.Wfuel'], 0.74714783, tol)#
        assert_near_equal(prob['TOC.inlet.F_ram'], 19867.69049170, tol)#
        assert_near_equal(prob['TOC.core_nozz.Fg'], 1558.85729859, tol)#
        assert_near_equal(prob['TOC.byp_nozz.Fg'], 24436.64223544, tol)#
        assert_near_equal(prob['TOC.perf.TSFC'], 0.4389403, tol)#
        assert_near_equal(prob['TOC.perf.OPR'], 53.63325, tol)#
        assert_near_equal(prob['TOC.balance.FAR'], 0.02652236, tol)#
        assert_near_equal(prob['TOC.hpc.Fl_O:tot:T'], 1530.39786420, tol)#
        
        assert_near_equal(prob['RTO.inlet.Fl_O:stat:W'], 1916.07027585, tol)#
        assert_near_equal(prob['RTO.inlet.Fl_O:tot:P'], 15.3028198, tol)#
        assert_near_equal(prob['RTO.hpc.Fl_O:tot:P'], 638.91726210, tol)#
        assert_near_equal(prob['RTO.burner.Wfuel'], 1.73490266, tol)#
        assert_near_equal(prob['RTO.inlet.F_ram'], 17048.01372698, tol)#
        assert_near_equal(prob['RTO.core_nozz.Fg'], 2224.0920195, tol)#
        assert_near_equal(prob['RTO.byp_nozz.Fg'], 37623.92170609, tol)#
        assert_near_equal(prob['RTO.perf.TSFC'], 0.27393200, tol)#
        assert_near_equal(prob['RTO.perf.OPR'], 41.75160332, tol)#
        assert_near_equal(prob['RTO.balance.FAR'], 0.02834445, tol)#
        assert_near_equal(prob['RTO.balance.fan_Nmech'], 2132.62333756, tol)#
        assert_near_equal(prob['RTO.balance.lp_Nmech'], 6611.18115904, tol)#
        assert_near_equal(prob['RTO.balance.hp_Nmech'], 22288.14292275, tol)#
        assert_near_equal(prob['RTO.hpc.Fl_O:tot:T'], 1720.90166426, tol)#
        
        assert_near_equal(prob['SLS.inlet.Fl_O:stat:W'], 1735.580099, tol)#
        assert_near_equal(prob['SLS.inlet.Fl_O:tot:P'], 14.62243072, tol)#
        assert_near_equal(prob['SLS.hpc.Fl_O:tot:P'], 522.96656034, tol)#
        assert_near_equal(prob['SLS.burner.Wfuel'], 1.32382416, tol)#
        assert_near_equal(prob['SLS.inlet.F_ram'], 61.76849298, tol)#
        assert_near_equal(prob['SLS.core_nozz.Fg'], 1542.39335422, tol)#
        assert_near_equal(prob['SLS.byp_nozz.Fg'], 27140.2151371, tol)#
        assert_near_equal(prob['SLS.perf.TSFC'], 0.16651387, tol)#
        assert_near_equal(prob['SLS.perf.OPR'], 35.76468033, tol)#
        assert_near_equal(prob['SLS.balance.FAR'], 0.02546042, tol)#
        assert_near_equal(prob['SLS.balance.fan_Nmech'], 1954.897347, tol)#
        assert_near_equal(prob['SLS.balance.lp_Nmech'], 6060.22652173, tol)#
        assert_near_equal(prob['SLS.balance.hp_Nmech'], 21600.83551971, tol)#
        assert_near_equal(prob['SLS.hpc.Fl_O:tot:T'], 1628.64528160, tol)#
        
        assert_near_equal(prob['CRZ.inlet.Fl_O:stat:W'], 802.81231779, tol)#
        assert_near_equal(prob['CRZ.inlet.Fl_O:tot:P'], 5.26210728, tol)#
        assert_near_equal(prob['CRZ.hpc.Fl_O:tot:P'], 264.63807022, tol)#
        assert_near_equal(prob['CRZ.burner.Wfuel'], 0.67580482, tol)#
        assert_near_equal(prob['CRZ.inlet.F_ram'], 19428.2645856, tol)#
        assert_near_equal(prob['CRZ.core_nozz.Fg'], 1386.10849325, tol)#
        assert_near_equal(prob['CRZ.byp_nozz.Fg'], 23557.16393820, tol)#
        assert_near_equal(prob['CRZ.perf.TSFC'], 0.44114123, tol)#
        assert_near_equal(prob['CRZ.perf.OPR'], 50.29127235, tol)#
        assert_near_equal(prob['CRZ.balance.FAR'], 0.02512285, tol)#    
        assert_near_equal(prob['CRZ.balance.fan_Nmech'], 2118.66425621, tol)#
        assert_near_equal(prob['CRZ.balance.lp_Nmech'], 6567.9076874, tol)#
        assert_near_equal(prob['CRZ.balance.hp_Nmech'], 20574.08585366, tol)#
        assert_near_equal(prob['CRZ.hpc.Fl_O:tot:T'], 1494.11501246, tol)#



if __name__ == "__main__":
    unittest.main()