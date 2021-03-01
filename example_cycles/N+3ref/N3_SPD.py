import numpy as np
import time
import pickle
from pprint import pprint

import openmdao.api as om

import pycycle.api as pyc

from N3ref import N3, viewer, MPN3

def N3_SPD_model():

    prob = om.Problem()

    prob.model = MPN3()    

    # setup the optimization
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.options['debug_print'] = ['desvars', 'nl_cons', 'objs']
    prob.driver.opt_settings={'Major step limit': 0.05}

    prob.model.add_design_var('fan:PRdes', lower=1.20, upper=1.4)
    prob.model.add_design_var('lpc:PRdes', lower=2.0, upper=4.0)
    prob.model.add_design_var('TOC.balance.rhs:hpc_PR', lower=40.0, upper=70.0, ref0=40.0, ref=70.0)
    prob.model.add_design_var('RTO_T4', lower=3000.0, upper=3600.0, ref0=3000.0, ref=3600.0)
    prob.model.add_design_var('T4_ratio.TR', lower=0.5, upper=0.95, ref0=0.5, ref=0.95)

    prob.model.add_objective('TOC.perf.TSFC')

    # to add the constraint to the model
    prob.model.add_constraint('TOC.fan_dia.FanDia', upper=100.0, ref=100.0)

    recorder = om.SqliteRecorder('N3_opt.sql')
    prob.model.add_recorder(recorder)
    prob.model.recording_options['record_inputs'] = True
    prob.model.recording_options['record_outputs'] = True

    return(prob)

if __name__ == "__main__":

    prob = N3_SPD_model()

    prob.setup()

    # prob.model.RTO.nonlinear_solver.options['maxiter'] = 0
    # prob.model.SLS.nonlinear_solver.options['maxiter'] = 0
    # prob.model.CRZ.nonlinear_solver.options['maxiter'] = 0
    # prob.model.nonlinear_solver.options['maxiter'] = 0

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
    W_guess = [1916.13, 1900. , 802.79]
    BPR_guess = [25.5620, 27.3467, 24.3233]
    hpc_PR_guess = [14., 14., 14.]
    fan_Nmech_guess = [2132.6, 1953.1, 2118.7]
    lp_Nmech_guess = [6611.2, 6054.5, 6567.9]
    hp_Nmech_guess = [22288.2, 21594.0, 20574.1]
    hpt_PR_guess = [4.210, 4.245, 4.197]
    lpt_PR_guess = [8.161, 8., 10.803]
    fan_Rline_guess = [1.7500, 1.7500, 1.9397]
    lpc_Rline_guess = [2.0052, 1.8632, 2.1075]
    hpc_Rline_guess = [2.0589, 2.0281, 1.9746]
    trq_guess = [52509.1, 41779.4, 22369.7]

    for i, pt in enumerate(prob.model.od_pts):

        # initial guesses
        prob[pt+'.balance.FAR'] = FAR_guess[i]
        prob[pt+'.balance.W'] = W_guess[i]
        prob[pt+'.balance.BPR'] = BPR_guess[i]
        prob[pt+'.balance.BPR'] = BPR_guess[i]
        prob[pt+'.balance.fan_Nmech'] = fan_Nmech_guess[i]
        prob[pt+'.balance.lp_Nmech'] = lp_Nmech_guess[i]
        prob[pt+'.balance.hp_Nmech'] = hp_Nmech_guess[i]
        prob[pt+'.hpc.PR'] = hpc_PR_guess[i]
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

    # prob.model.RTO.list_outputs(residuals=True, prom_name=True)
    # prob.model.RTO.hpc.ideal_flow.list_inputs(units=True, prom_name=True)
    # exit()

    for pt in ['TOC']+prob.model.od_pts:
        viewer(prob, pt)

    print("time", time.time() - st)