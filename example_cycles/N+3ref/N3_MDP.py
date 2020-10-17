import numpy as np
import time
import pickle
from pprint import pprint

import openmdao.api as om

import pycycle.api as pyc

from N3ref import N3, viewer, MPN3

def N3_MDP_model():

    prob = om.Problem()

    prob.model = MPN3(order_add=['bal'])

    bal = prob.model.add_subsystem('bal', om.BalanceComp(), promotes=['RTO_T4',])
    bal.add_balance('TOC_BPR', val=23.7281, units=None, eq_units='ft/s', use_mult=True)
    prob.model.connect('bal.TOC_BPR', 'TOC.splitter.BPR')
    prob.model.connect('CRZ.byp_nozz.Fl_O:stat:V', 'bal.lhs:TOC_BPR')
    prob.model.connect('CRZ.core_nozz.Fl_O:stat:V', 'bal.rhs:TOC_BPR')

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

    # setup the optimization
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['debug_print'] = ['desvars', 'nl_cons', 'objs']
    prob.driver.opt_settings={'Major step limit': 0.05}

    prob.model.add_design_var('fan:PRdes', lower=1.20, upper=1.4)
    prob.model.add_design_var('lpc:PRdes', lower=2.0, upper=4.0)
    prob.model.add_design_var('TOC.balance.rhs:hpc_PR', lower=40.0, upper=70.0, ref0=40.0, ref=70.0)
    prob.model.add_design_var('RTO_T4', lower=3000.0, upper=3600.0, ref0=3000.0, ref=3600.0)
    prob.model.add_design_var('bal.mult:TOC_BPR', lower=1.35, upper=1.45, ref0=1.35, ref=1.45)
    prob.model.add_design_var('T4_ratio.TR', lower=0.5, upper=0.95, ref0=0.5, ref=0.95)

    prob.model.add_objective('TOC.perf.TSFC')

    prob.model.add_constraint('TOC.fan_dia.FanDia', upper=100.0, ref=100.0)

    recorder = om.SqliteRecorder('N3_opt.sql')
    prob.model.add_recorder(recorder)
    prob.model.recording_options['record_inputs'] = True
    prob.model.recording_options['record_outputs'] = True

    prob.model.set_input_defaults('RTO_T4', 3400.0, units='degR')

    return(prob)

if __name__ == "__main__":

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

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()

    for pt in ['TOC']+prob.model.od_pts:
        viewer(prob, pt)

    print()
    print('Diameter', prob['TOC.fan_dia.FanDia'][0])
    print("time", time.time() - st)