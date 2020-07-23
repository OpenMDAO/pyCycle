import numpy as np
import time
import pickle
from pprint import pprint

from openmdao.api import DirectSolver, BoundsEnforceLS, NewtonSolver, ArmijoGoldsteinLS, LinearBlockGS, pyOptSparseDriver
from openmdao.api import Problem, IndepVarComp, SqliteRecorder, CaseReader, BalanceComp, ScipyKrylov, PETScKrylov, ExecComp
from openmdao.utils.units import convert_units as cu

import pycycle.api as pyc

from N3ref import N3, viewer, MPN3

def N3_MDP_verif_model(OD_statics=True):

    prob = Problem()

    prob.model = MPN3(order_add=['bal'], statics=OD_statics)

    prob.model.pyc_add_cycle_param('ext_ratio.core_Cv', 0.9999)
    prob.model.pyc_add_cycle_param('ext_ratio.byp_Cv', 0.9975)

    bal = prob.model.add_subsystem('bal', BalanceComp(), promotes_inputs=['RTO_T4',])

    bal.add_balance('TOC_BPR', val=23.7281, units=None, eq_units=None)
    prob.model.connect('bal.TOC_BPR', 'TOC.splitter.BPR')
    prob.model.connect('CRZ.ext_ratio.ER', 'bal.lhs:TOC_BPR')

    bal.add_balance('TOC_W', val=820.95, units='lbm/s', eq_units='degR', rhs_name='RTO_T4')
    prob.model.connect('bal.TOC_W', 'TOC.fc.W')
    prob.model.connect('RTO.burner.Fl_O:tot:T', 'bal.lhs:TOC_W')

    bal.add_balance('CRZ_Fn_target', val=5514.4, units='lbf', eq_units='lbf', use_mult=True, mult_val=0.9, ref0=5000.0, ref=7000.0)
    prob.model.connect('bal.CRZ_Fn_target', 'CRZ.balance.rhs:FAR')
    prob.model.connect('TOC.perf.Fn', 'bal.lhs:CRZ_Fn_target')
    prob.model.connect('CRZ.perf.Fn','bal.rhs:CRZ_Fn_target')

    bal.add_balance('SLS_Fn_target', val=28620.8, units='lbf', eq_units='lbf', use_mult=True, mult_val=1.2553, ref0=28000.0, ref=30000.0)
    prob.model.connect('bal.SLS_Fn_target', 'SLS.balance.rhs:FAR')
    prob.model.connect('RTO.perf.Fn', 'bal.lhs:SLS_Fn_target')
    prob.model.connect('SLS.perf.Fn','bal.rhs:SLS_Fn_target')

    prob.model.set_input_defaults('RTO_T4', 3400.0, units='degR')

    return(prob)

if __name__ == "__main__":

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

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()

    data = prob.compute_totals(of=['TOC.perf.Fn','RTO.perf.Fn','SLS.perf.Fn','CRZ.perf.Fn',
                                'TOC.perf.TSFC','RTO.perf.TSFC','SLS.perf.TSFC','CRZ.perf.TSFC',
                                # 'bal.TOC_BPR','TOC.hpc_CS.CS',], wrt=['OPR', 'RTO_T4'])
                                'TOC.hpc_CS.CS',], wrt=['TOC.balance.rhs:hpc_PR', 'RTO_T4'])
    pprint(data)

    with open('derivs.pkl','wb') as f:
        pickle.dump(data, file=f)

    for pt in ['TOC']+prob.model.od_pts:
        viewer(prob, pt)

    print()
    print('Diameter', prob['TOC.fan_dia.FanDia'][0])
    print('ER', prob['CRZ.ext_ratio.ER'])
    print("time", time.time() - st)