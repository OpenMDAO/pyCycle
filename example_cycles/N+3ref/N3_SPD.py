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
    prob.setup(check=False)

    pts = ['RTO','SLS','CRZ']

    prob.set_val('TOC.splitter.BPR', 23.94514401), 
    prob.set_val('fan:PRdes', 1.300),
    prob.set_val('lpc:PRdes', 3.000),
    prob.set_val('TOC.balance.rhs:hpc_PR', 53.6332)
    prob.set_val('T4_ratio.TR', 0.926470588)
    prob.set_val('RTO_T4', 3400.0, units='degR')
    prob.set_val('SLS.balance.rhs:FAR', 28620.84, units='lbf')
    prob.set_val('CRZ.balance.rhs:FAR', 5510.72833567, units='lbf') 
    prob.set_val('TOC.fc.W', 820.44097898, units='lbm/s')

    prob['RTO.hpt_cooling.x_factor'] = 0.9

    # initial guesses
    prob['TOC.balance.FAR'] = 0.02650
    prob['TOC.balance.lpt_PR'] = 10.937
    prob['TOC.balance.hpt_PR'] = 4.185
    prob['TOC.fc.balance.Pt'] = 5.272
    prob['TOC.fc.balance.Tt'] = 444.41

    for pt in pts:

        if pt == 'RTO':
            prob[pt+'.balance.FAR'] = 0.02832
            prob[pt+'.balance.W'] = 1916.13
            prob[pt+'.balance.BPR'] = 25.5620
            prob[pt+'.balance.fan_Nmech'] = 2132.6
            prob[pt+'.balance.lp_Nmech'] = 6611.2
            prob[pt+'.balance.hp_Nmech'] = 22288.2
            prob[pt+'.fc.balance.Pt'] = 15.349
            prob[pt+'.fc.balance.Tt'] = 552.49
            prob[pt+'.hpt.PR'] = 4.210
            prob[pt+'.lpt.PR'] = 8.161
            prob[pt+'.fan.map.RlineMap'] = 1.7500
            prob[pt+'.lpc.map.RlineMap'] = 2.0052
            prob[pt+'.hpc.map.RlineMap'] = 2.0589
            prob[pt+'.gearbox.trq_base'] = 52509.1

        if pt == 'SLS':
            prob[pt+'.balance.FAR'] = 0.02541
            prob[pt+'.balance.W'] = 2000. #1734.44
            prob[pt+'.balance.BPR'] = 27.3467
            prob[pt+'.balance.fan_Nmech'] = 1953.1
            prob[pt+'.balance.lp_Nmech'] = 6054.5
            prob[pt+'.balance.hp_Nmech'] = 21594.0
            prob[pt+'.fc.balance.Pt'] = 14.696
            prob[pt+'.fc.balance.Tt'] = 545.67
            prob[pt+'.hpt.PR'] = 4.245
            prob[pt+'.lpt.PR'] = 7.001
            prob[pt+'.fan.map.RlineMap'] = 1.7500
            prob[pt+'.lpc.map.RlineMap'] = 1.8632
            prob[pt+'.hpc.map.RlineMap'] = 2.0281
            prob[pt+'.gearbox.trq_base'] = 41779.4

        if pt == 'CRZ':
            prob[pt+'.balance.FAR'] = 0.02510
            prob[pt+'.balance.W'] = 802.79
            prob[pt+'.balance.BPR'] = 24.3233
            prob[pt+'.balance.fan_Nmech'] = 2118.7
            prob[pt+'.balance.lp_Nmech'] = 6567.9
            prob[pt+'.balance.hp_Nmech'] = 20574.1
            prob[pt+'.fc.balance.Pt'] = 5.272
            prob[pt+'.fc.balance.Tt'] = 444.41
            prob[pt+'.hpt.PR'] = 4.197
            prob[pt+'.lpt.PR'] = 10.803
            prob[pt+'.fan.map.RlineMap'] = 1.9397
            prob[pt+'.lpc.map.RlineMap'] = 2.1075
            prob[pt+'.hpc.map.RlineMap'] = 1.9746
            prob[pt+'.gearbox.trq_base'] = 22369.7



    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()

    for pt in ['TOC']+pts:
        viewer(prob, pt)

    print("time", time.time() - st)

    print('TOC')
    print(prob['TOC.inlet.Fl_O:stat:W'] - 820.44097898)#
    print(prob['TOC.inlet.Fl_O:tot:P'] - 5.26210728)#
    print(prob['TOC.hpc.Fl_O:tot:P'] - 275.21039426)#
    print(prob['TOC.burner.Wfuel'] - 0.74668441)#
    print(prob['TOC.inlet.F_ram'] - 19854.88340967)#
    print(prob['TOC.core_nozz.Fg'] - 1547.13847348)#
    print(prob['TOC.byp_nozz.Fg'] - 24430.7872167)#
    print(prob['TOC.perf.TSFC'] - 0.43900789)#
    print(prob['TOC.perf.OPR'] - 52.30041498)#
    print(prob['TOC.balance.FAR'] - 0.02669913)#
    print(prob['TOC.hpc.Fl_O:tot:T'] - 1517.98001185)#
    print('............................')
    print('RTO')
    print(prob['RTO.inlet.Fl_O:stat:W'] - 1915.22687254)#
    print(prob['RTO.inlet.Fl_O:tot:P'] - 15.3028198)#
    print(prob['RTO.hpc.Fl_O:tot:P'] - 623.4047562)#
    print(prob['RTO.burner.Wfuel'] - 1.73500397)#
    print(prob['RTO.inlet.F_ram'] - 17040.51023484)#
    print(prob['RTO.core_nozz.Fg'] - 2208.78618847)#
    print(prob['RTO.byp_nozz.Fg'] - 37631.72404627)#
    print(prob['RTO.perf.TSFC'] - 0.273948)#
    print(prob['RTO.perf.OPR'] - 40.73790087)#
    print(prob['RTO.balance.FAR'] - 0.02853677)#
    print(prob['RTO.balance.fan_Nmech'] - 2133.2082055)#
    print(prob['RTO.balance.lp_Nmech'] - 6612.99426306)#
    print(prob['RTO.balance.hp_Nmech'] - 22294.4214065)#
    print(prob['RTO.hpc.Fl_O:tot:T'] - 1707.8424979)#
    print('............................')
    print('SLS')
    print(prob['SLS.inlet.Fl_O:stat:W'] - 1733.66974646)#
    print(prob['SLS.inlet.Fl_O:tot:P'] - 14.62242048)#
    print(prob['SLS.hpc.Fl_O:tot:P'] - 509.33522171)#
    print(prob['SLS.burner.Wfuel'] - 1.3201035)#
    print(prob['SLS.inlet.F_ram'] - 0.06170051)#
    print(prob['SLS.core_nozz.Fg'] - 1526.45701492)#
    print(prob['SLS.byp_nozz.Fg'] - 27094.44468547)#
    print(prob['SLS.perf.TSFC'] - 0.16604588)#
    print(prob['SLS.perf.OPR'] - 34.8324836)#
    print(prob['SLS.balance.FAR'] - 0.02559136)#
    print(prob['SLS.balance.fan_Nmech'] - 1953.67749381)#
    print(prob['SLS.balance.lp_Nmech'] - 6056.44494761)#
    print(prob['SLS.balance.hp_Nmech'] - 21599.4239832)#
    print(prob['SLS.hpc.Fl_O:tot:T'] - 1615.20710655)#
    print('............................')
    print('CRZ')
    print(prob['CRZ.inlet.Fl_O:stat:W'] - 802.28669491)#
    print(prob['CRZ.inlet.Fl_O:tot:P'] - 5.26210728)#
    print(prob['CRZ.hpc.Fl_O:tot:P'] - 258.04394098)#
    print(prob['CRZ.burner.Wfuel'] - 0.67533764)#
    print(prob['CRZ.inlet.F_ram'] - 19415.54505032)#
    print(prob['CRZ.core_nozz.Fg'] - 1375.4427785)#
    print(prob['CRZ.byp_nozz.Fg'] - 23550.83060746)#
    print(prob['CRZ.perf.TSFC'] - 0.44117862)#
    print(prob['CRZ.perf.OPR'] - 49.03813765)#
    print(prob['CRZ.balance.FAR'] - 0.02528878)#
    print(prob['CRZ.balance.fan_Nmech'] - 2118.62554023)#
    print(prob['CRZ.balance.lp_Nmech'] - 6567.78766693)#
    print(prob['CRZ.balance.hp_Nmech'] - 20574.43651756)#
    print(prob['CRZ.hpc.Fl_O:tot:T'] - 1481.9756247)#   