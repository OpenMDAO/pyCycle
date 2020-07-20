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

    # prob.model.linear_solver = PETScKrylov()
    # prob.model.linear_solver.options['iprint'] = 2
    # prob.model.linear_solver.precon = DirectSolver()
    # prob.model.jacobian = CSCJacobian()

    # prob.model.linear_solver = PETScKrylov()
    # prob.model.linear_solver.options['iprint'] = 2
    # prob.model.linear_solver.options['atol'] = 1e-6
    # prob.model.linear_solver.precon = LinearBlockGS()
    # prob.model.linear_solver.precon.options['maxiter'] = 2
    # prob.model.jacobian = CSCJacobian()

    ###############
    #BROKEN!!!!
    ##############
    # prob.model.linear_solver = ScipyKrylov()
    # prob.model.linear_solver.options['iprint'] = 2
    # prob.model.linear_solver.precon = LinearBlockGS()
    # prob.model.linear_solver.precon = LinearRunOnce()
    ####################

    # prob.model.linear_solver = LinearBlockGS()
    # prob.model.linear_solver.options['maxiter'] = 10
    # prob.model.linear_solver.options['iprint'] = 2


    prob.model.set_input_defaults('RTO_T4', 3400.0, units='degR')

    return(prob)

if __name__ == "__main__":

    OD_statics = True
    prob = N3_MDP_verif_model(OD_statics)
    prob.setup(check=False)
    pts = ['RTO','SLS','CRZ']

    prob.set_val('TOC.splitter.BPR', 23.7281)
    prob.set_val('TOC.balance.rhs:hpc_PR', 55.0)
    prob.set_val('SLS.fc.MN', 0.001)
    prob.set_val('SLS.balance.rhs:FAR', 28620.9, units='lbf')
    prob.set_val('CRZ.balance.rhs:FAR', 5466.5, units='lbf')
    prob.set_val('bal.rhs:TOC_BPR', 1.40)
    prob.set_val('fan:PRdes', 1.300)
    prob.set_val('TOC.opr_calc.FPR', 1.300)
    prob.set_val('lpc:PRdes', 3.000),
    prob.set_val('TOC.opr_calc.LPCPR', 3.000)
    prob.set_val('T4_ratio.TR', 0.926470588)

    prob['RTO.hpt_cooling.x_factor'] = 0.9

    # initial guesses
    prob['TOC.balance.FAR'] = 0.02650
    prob['bal.TOC_W'] = 820.95
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
            prob[pt+'.balance.W'] = 1734.44
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

    data = prob.compute_totals(of=['TOC.perf.Fn','RTO.perf.Fn','SLS.perf.Fn','CRZ.perf.Fn',
                                'TOC.perf.TSFC','RTO.perf.TSFC','SLS.perf.TSFC','CRZ.perf.TSFC',
                                # 'bal.TOC_BPR','TOC.hpc_CS.CS',], wrt=['OPR', 'RTO_T4'])
                                'TOC.hpc_CS.CS',], wrt=['TOC.balance.rhs:hpc_PR', 'RTO_T4'])
    pprint(data)

    with open('derivs.pkl','wb') as f:
        pickle.dump(data, file=f)
    
    # ##################################
    # # Check Totals: Hand calcualted values
    # #   'TOC.perf.TSFC' wrt 'des_vars.OPR' = -1.87E-4
    # #   'TOC.perf.TSFC' wrt 'des_vars.RTO:T4max' = 6.338e-6
    # # prob.check_totals(step_calc='rel', step=1e-3)
    # ##################################

    # print('OPR', prob['fan:PRdes']*prob['lpc:PRdes']*prob['TOC.balance.hpc_PR'])
    # print('OPR', prob['OPR'])
    # print('T4max', prob['RTO:T4max'])
    # print('T4max', prob['T4_ratio.TOC_T4'])
    # print('TSFC', prob['TOC.perf.TSFC'])
    # print('Fn', prob['TOC.perf.Fn'])

    # prob['RTO:T4max'] *= (1.0+1e-4)
    # # prob['OPR'] *= (1.0+1e-4)
    # prob.run_model()

    # print('OPR', prob['fan:PRdes']*prob['lpc:PRdes']*prob['TOC.balance.hpc_PR'])
    # print('OPR', prob['OPR'])
    # print('T4max', prob['RTO:T4max'])
    # print('T4max', prob['T4_ratio.TOC_T4'])
    # print('TSFC', prob['TOC.perf.TSFC'])
    # print('Fn', prob['TOC.perf.Fn'])

    # prob['RTO:T4max'] /= (1.0+1e-4)
    # prob['OPR'] *= (1.0+1e-4)
    # prob.run_model()

    # print('OPR', prob['fan:PRdes']*prob['lpc:PRdes']*prob['TOC.balance.hpc_PR'])
    # print('OPR', prob['OPR'])
    # print('T4max', prob['RTO:T4max'])
    # print('T4max', prob['T4_ratio.TOC_T4'])
    # print('TSFC', prob['TOC.perf.TSFC'])
    # print('Fn', prob['TOC.perf.Fn'])


    # exit()

    # prob.model.list_outputs(residuals=True)


    # prob['OPR'] = 65.0
    # prob['RTO:T4max'] = 3150.0
    # prob.run_model()

    # prob['OPR'] = 70.0
    # prob['RTO:T4max'] = 3373.2184409
    # prob.run_model()


    for pt in ['TOC']+pts:
        viewer(prob, pt)

    print()
    print('Diameter', prob['TOC.fan_dia.FanDia'][0])
    print('ER', prob['CRZ.ext_ratio.ER'])
    print("time", time.time() - st)

    prob.model.list_outputs(explicit=True, residuals=True, residuals_tol=1e-6)

    print('TOC')
    print(prob['TOC.inlet.Fl_O:stat:W'] - 820.92037027)#
    print(prob['TOC.inlet.Fl_O:tot:P'] - 5.26210728)#
    print(prob['TOC.hpc.Fl_O:tot:P'] - 282.22391512)#
    print(prob['TOC.burner.Wfuel'] - 0.74642969)#
    print(prob['TOC.inlet.F_ram'] - 19866.48480269)#
    print(prob['TOC.core_nozz.Fg'] - 1556.44941177)#
    print(prob['TOC.byp_nozz.Fg'] - 24436.69962673)#
    print(prob['TOC.perf.TSFC'] - 0.43859869)#
    print(prob['TOC.perf.OPR'] - 53.63325)#
    print(prob['TOC.balance.FAR'] - 0.02650755)#
    print(prob['TOC.hpc.Fl_O:tot:T'] - 1530.58386828)#
    print('............................')
    print('RTO')
    print(prob['RTO.inlet.Fl_O:stat:W'] - 1916.01614631)#
    print(prob['RTO.inlet.Fl_O:tot:P'] - 15.3028198)#
    print(prob['RTO.hpc.Fl_O:tot:P'] - 638.95720683)#
    print(prob['RTO.burner.Wfuel'] - 1.73329552)#
    print(prob['RTO.inlet.F_ram'] - 17047.53270726)#
    print(prob['RTO.core_nozz.Fg'] - 2220.78852731)#
    print(prob['RTO.byp_nozz.Fg'] - 37626.74417995)#
    print(prob['RTO.perf.TSFC'] - 0.27367824)#
    print(prob['RTO.perf.OPR'] - 41.7542136)#
    print(prob['RTO.balance.FAR'] - 0.02832782)#
    print(prob['RTO.balance.fan_Nmech'] - 2132.71615737)#
    print(prob['RTO.balance.lp_Nmech'] - 6611.46890258)#
    print(prob['RTO.balance.hp_Nmech'] - 22288.52228766)#
    print(prob['RTO.hpc.Fl_O:tot:T'] - 1721.14599533)#
    print('............................')
    print('SLS')
    print(prob['SLS.inlet.Fl_O:stat:W'] - 1735.52737576)#
    print(prob['SLS.inlet.Fl_O:tot:P'] - 14.62243072)#
    print(prob['SLS.hpc.Fl_O:tot:P'] - 522.99027178)#
    print(prob['SLS.burner.Wfuel'] - 1.32250739)#
    print(prob['SLS.inlet.F_ram'] - 61.76661874)#
    print(prob['SLS.core_nozz.Fg'] - 1539.99663978)#
    print(prob['SLS.byp_nozz.Fg'] - 27142.60997896)#
    print(prob['SLS.perf.TSFC'] - 0.16634825)#
    print(prob['SLS.perf.OPR'] - 35.76630191)#
    print(prob['SLS.balance.FAR'] - 0.02544378)#
    print(prob['SLS.balance.fan_Nmech'] - 1954.97855672)#
    print(prob['SLS.balance.lp_Nmech'] - 6060.47827242)#
    print(prob['SLS.balance.hp_Nmech'] - 21601.07508077)#
    print(prob['SLS.hpc.Fl_O:tot:T'] - 1628.85845903)#
    print('............................')
    print('CRZ')
    print(prob['CRZ.inlet.Fl_O:stat:W'] - 802.76200548)#
    print(prob['CRZ.inlet.Fl_O:tot:P'] - 5.26210728)#
    print(prob['CRZ.hpc.Fl_O:tot:P'] - 264.63649163)#
    print(prob['CRZ.burner.Wfuel'] - 0.67514702)#
    print(prob['CRZ.inlet.F_ram'] - 19427.04768877)#
    print(prob['CRZ.core_nozz.Fg'] - 1383.93102366)#
    print(prob['CRZ.byp_nozz.Fg'] - 23557.11447723)#
    print(prob['CRZ.perf.TSFC'] - 0.44079257)#
    print(prob['CRZ.perf.OPR'] - 50.29097236)#
    print(prob['CRZ.balance.FAR'] - 0.02510864)#    
    print(prob['CRZ.balance.fan_Nmech'] - 2118.65676797)#
    print(prob['CRZ.balance.lp_Nmech'] - 6567.88447364)#
    print(prob['CRZ.balance.hp_Nmech'] - 20574.08438737)#
    print(prob['CRZ.hpc.Fl_O:tot:T'] - 1494.29261337)#
