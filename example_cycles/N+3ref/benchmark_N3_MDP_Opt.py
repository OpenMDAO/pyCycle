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

        tol = 1e-4

        assert_near_equal(prob['TOC.inlet.Fl_O:stat:W'], 810.91780825, tol)#
        assert_near_equal(prob['TOC.inlet.Fl_O:tot:P'], 5.26210728, tol)#
        assert_near_equal(prob['TOC.hpc.Fl_O:tot:P'], 359.19407379, tol)#
        assert_near_equal(prob['TOC.burner.Wfuel'], 0.84696398, tol)#
        assert_near_equal(prob['TOC.inlet.F_ram'], 19624.42022041, tol)#
        assert_near_equal(prob['TOC.core_nozz.Fg'], 2062.17720394, tol)#
        assert_near_equal(prob['TOC.byp_nozz.Fg'], 24607.96791651, tol)#
        assert_near_equal(prob['TOC.perf.TSFC'], 0.43275466, tol)#
        assert_near_equal(prob['TOC.perf.OPR'], 68.2605, tol)#
        assert_near_equal(prob['TOC.balance.FAR'], 0.02157518, tol)#
        assert_near_equal(prob['TOC.hpc.Fl_O:tot:T'], 1629.57914093, tol)#
        assert_near_equal(prob['RTO.inlet.Fl_O:stat:W'], 1836.24241256, tol)#
        assert_near_equal(prob['RTO.inlet.Fl_O:tot:P'], 15.3028198, tol)#
        assert_near_equal(prob['RTO.hpc.Fl_O:tot:P'], 746.29459765, tol)#
        assert_near_equal(prob['RTO.burner.Wfuel'], 1.74761057, tol)#
        assert_near_equal(prob['RTO.inlet.F_ram'], 16337.75511063, tol)#
        assert_near_equal(prob['RTO.core_nozz.Fg'], 2531.69432791, tol)#
        assert_near_equal(prob['RTO.byp_nozz.Fg'], 36606.06077695, tol)#
        assert_near_equal(prob['RTO.perf.TSFC'], 0.27593851, tol)#
        assert_near_equal(prob['RTO.perf.OPR'], 48.76843661, tol)#
        assert_near_equal(prob['RTO.balance.FAR'], 0.02201472, tol)#
        assert_near_equal(prob['RTO.balance.fan_Nmech'], 2047.49245139, tol)#
        assert_near_equal(prob['RTO.balance.lp_Nmech'], 6347.27346339, tol)#
        assert_near_equal(prob['RTO.balance.hp_Nmech'], 21956.80317383, tol)#
        assert_near_equal(prob['RTO.hpc.Fl_O:tot:T'], 1785.68517839, tol)#
        assert_near_equal(prob['SLS.inlet.Fl_O:stat:W'], 1665.15413784, tol)#
        assert_near_equal(prob['SLS.inlet.Fl_O:tot:P'], 14.62243072, tol)#
        assert_near_equal(prob['SLS.hpc.Fl_O:tot:P'], 618.67644706, tol)#
        assert_near_equal(prob['SLS.burner.Wfuel'], 1.3646179, tol)#
        assert_near_equal(prob['SLS.inlet.F_ram'], 59.26206766, tol)#
        assert_near_equal(prob['SLS.core_nozz.Fg'], 1803.28636041, tol)#
        assert_near_equal(prob['SLS.byp_nozz.Fg'], 26876.81570145, tol)#
        assert_near_equal(prob['SLS.perf.TSFC'], 0.17164501, tol)#
        assert_near_equal(prob['SLS.perf.OPR'], 42.31009597, tol)#
        assert_near_equal(prob['SLS.balance.FAR'], 0.02009085, tol)#
        assert_near_equal(prob['SLS.balance.fan_Nmech'], 1885.03384389, tol)#
        assert_near_equal(prob['SLS.balance.lp_Nmech'], 5843.64806172, tol)#
        assert_near_equal(prob['SLS.balance.hp_Nmech'], 21335.89542087, tol)#
        assert_near_equal(prob['SLS.hpc.Fl_O:tot:T'], 1698.10451317, tol)#
        assert_near_equal(prob['CRZ.inlet.Fl_O:stat:W'], 792.88599794, tol)#
        assert_near_equal(prob['CRZ.inlet.Fl_O:tot:P'], 5.26210728, tol)#
        assert_near_equal(prob['CRZ.hpc.Fl_O:tot:P'], 336.48275934, tol)#
        assert_near_equal(prob['CRZ.burner.Wfuel'], 0.76513434, tol)#
        assert_near_equal(prob['CRZ.inlet.F_ram'], 19188.04575789, tol)#
        assert_near_equal(prob['CRZ.core_nozz.Fg'], 1832.22026473, tol)#
        assert_near_equal(prob['CRZ.byp_nozz.Fg'], 23696.97790339, tol)#
        assert_near_equal(prob['CRZ.perf.TSFC'], 0.43438218, tol)#
        assert_near_equal(prob['CRZ.perf.OPR'], 63.9444887, tol)#
        assert_near_equal(prob['CRZ.balance.FAR'], 0.02044895, tol)#
        assert_near_equal(prob['CRZ.balance.fan_Nmech'], 2118.19810925, tol)#
        assert_near_equal(prob['CRZ.balance.lp_Nmech'], 6566.46262111, tol)#
        assert_near_equal(prob['CRZ.balance.hp_Nmech'], 20575.59530495, tol)#
        assert_near_equal(prob['CRZ.hpc.Fl_O:tot:T'], 1591.47697124, tol)#


if __name__ == "__main__":
    unittest.main()