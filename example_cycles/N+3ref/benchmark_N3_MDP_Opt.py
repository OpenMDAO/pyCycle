import numpy as np 
import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import pycycle.api as pyc

from N3_MDP_Opt import N3_MDP_Opt_model

class N3MDPOptTestCase(unittest.TestCase):

    def benchmark_case1(self):

        prob = N3_MDP_Opt_model()

        prob.setup()

        # Define the design point
        prob.set_val('TOC.splitter.BPR', 23.7281)
        prob.set_val('TOC.balance.rhs:hpc_PR', 55.0)

        # Set specific cycle parameters
        prob.set_val('SLS.fc.MN', 0.001)
        prob.set_val('SLS.balance.rhs:FAR', 28620.9, units='lbf')
        prob.set_val('CRZ.balance.rhs:FAR', 5466.5, units='lbf')
        prob.set_val('bal.rhs:TOC_BPR', 1.40)
        prob.set_val('T4_ratio.TR', 0.926470588)
        prob.set_val('fan:PRdes', 1.300)
        prob.set_val('lpc:PRdes', 3.000)
        prob.set_val('RTO.hpt_cooling.x_factor', 0.9)

        # Set inital guesses for balances
        prob['TOC.balance.FAR'] = 0.02650
        prob['bal.TOC_W'] = 820.95
        prob['TOC.balance.lpt_PR'] = 10.937
        prob['TOC.balance.hpt_PR'] = 4.185
        prob['TOC.fc.balance.Pt'] = 5.272
        prob['TOC.fc.balance.Tt'] = 444.41

        FAR_guess = [0.02832, 0.02541, 0.02510]
        W_guess = [1916.13, 2000 , 802.79]
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

        prob.run_driver()

        tol = 5e-4

        assert_near_equal(prob['TOC.inlet.Fl_O:stat:W'], 810.917847, tol)#
        assert_near_equal(prob['TOC.inlet.Fl_O:tot:P'], 5.26210728, tol)#
        assert_near_equal(prob['TOC.hpc.Fl_O:tot:P'], 359.19407379, tol)#
        assert_near_equal(prob['TOC.burner.Wfuel'], 0.84806746, tol)#
        assert_near_equal(prob['TOC.inlet.F_ram'], 19624.42047734, tol)#
        assert_near_equal(prob['TOC.core_nozz.Fg'], 2070.8761044, tol)#
        assert_near_equal(prob['TOC.byp_nozz.Fg'], 24602.79441880, tol)#
        assert_near_equal(prob['TOC.perf.TSFC'], 0.43310166, tol)#
        assert_near_equal(prob['TOC.perf.OPR'], 68.2605, tol)#
        assert_near_equal(prob['TOC.balance.FAR'], 0.02151082, tol)#
        assert_near_equal(prob['TOC.hpc.Fl_O:tot:T'], 1629.1546998, tol)#
        
        assert_near_equal(prob['RTO.inlet.Fl_O:stat:W'], 1836.12078846, tol)#
        assert_near_equal(prob['RTO.inlet.Fl_O:tot:P'], 15.3028198, tol)#
        assert_near_equal(prob['RTO.hpc.Fl_O:tot:P'], 746.17260179, tol)#
        assert_near_equal(prob['RTO.burner.Wfuel'], 1.74940970, tol)#
        assert_near_equal(prob['RTO.inlet.F_ram'], 16336.67240732, tol)#
        assert_near_equal(prob['RTO.core_nozz.Fg'], 2541.90947796, tol)#
        assert_near_equal(prob['RTO.byp_nozz.Fg'], 36594.76292895, tol)#
        assert_near_equal(prob['RTO.perf.TSFC'], 0.27622258, tol)#
        assert_near_equal(prob['RTO.perf.OPR'], 48.7604645, tol)#
        assert_near_equal(prob['RTO.balance.FAR'], 0.02194653, tol)#
        assert_near_equal(prob['RTO.balance.fan_Nmech'], 2047.30894830, tol)#
        assert_near_equal(prob['RTO.balance.lp_Nmech'], 6346.70459964, tol)#
        assert_near_equal(prob['RTO.balance.hp_Nmech'], 21956.08415951, tol)#
        assert_near_equal(prob['RTO.hpc.Fl_O:tot:T'], 1785.14530857, tol)#
        
        assert_near_equal(prob['SLS.inlet.Fl_O:stat:W'], 1665.04070941, tol)#
        assert_near_equal(prob['SLS.inlet.Fl_O:tot:P'], 14.62243072, tol)#
        assert_near_equal(prob['SLS.hpc.Fl_O:tot:P'], 618.61710206, tol)#
        assert_near_equal(prob['SLS.burner.Wfuel'], 1.36618966, tol)#
        assert_near_equal(prob['SLS.inlet.F_ram'], 59.25802874, tol)#
        assert_near_equal(prob['SLS.core_nozz.Fg'], 1810.85872520, tol)#
        assert_near_equal(prob['SLS.byp_nozz.Fg'], 26869.23930311, tol)#
        assert_near_equal(prob['SLS.perf.TSFC'], 0.17184271, tol)#
        assert_near_equal(prob['SLS.perf.OPR'], 42.3060375, tol)#
        assert_near_equal(prob['SLS.balance.FAR'], 0.02003034, tol)#
        assert_near_equal(prob['SLS.balance.fan_Nmech'], 1884.87786556, tol)#
        assert_near_equal(prob['SLS.balance.lp_Nmech'], 5843.16452534, tol)#
        assert_near_equal(prob['SLS.balance.hp_Nmech'], 21335.53413848, tol)#
        assert_near_equal(prob['SLS.hpc.Fl_O:tot:T'], 1697.63532613, tol)#
        
        assert_near_equal(prob['CRZ.inlet.Fl_O:stat:W'], 792.89092687, tol)#
        assert_near_equal(prob['CRZ.inlet.Fl_O:tot:P'], 5.26210728, tol)#
        assert_near_equal(prob['CRZ.hpc.Fl_O:tot:P'], 336.49153292, tol)#
        assert_near_equal(prob['CRZ.burner.Wfuel'], 0.76613799, tol)#
        assert_near_equal(prob['CRZ.inlet.F_ram'], 19188.16437330, tol)#
        assert_near_equal(prob['CRZ.core_nozz.Fg'], 1840.07482549, tol)#
        assert_near_equal(prob['CRZ.byp_nozz.Fg'], 23692.4166250, tol)#
        assert_near_equal(prob['CRZ.perf.TSFC'], 0.43473433, tol)#
        assert_near_equal(prob['CRZ.perf.OPR'], 63.94615602, tol)#
        assert_near_equal(prob['CRZ.balance.FAR'], 0.0203877, tol)#
        assert_near_equal(prob['CRZ.balance.fan_Nmech'], 2118.22224123, tol)#
        assert_near_equal(prob['CRZ.balance.lp_Nmech'], 6566.53743080, tol)#
        assert_near_equal(prob['CRZ.balance.hp_Nmech'], 20575.73146332, tol)#
        assert_near_equal(prob['CRZ.hpc.Fl_O:tot:T'], 1591.08096223, tol)#



if __name__ == "__main__":
    unittest.main()