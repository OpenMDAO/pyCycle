import numpy as np
import time

from openmdao.api import DirectSolver, BoundsEnforceLS, NewtonSolver, ArmijoGoldsteinLS, LinearBlockGS
from openmdao.api import Problem, IndepVarComp, SqliteRecorder, CaseReader, BalanceComp, ScipyKrylov
from openmdao.utils.units import convert_units as cu

from pycycle.elements.api import CombineCooling, TurbineCooling
from pycycle.cea import species_data
from pycycle.connect_flow import connect_flow

from N3ref import N3, viewer

prob = Problem()

des_vars = prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=["*"])

des_vars.add_output('inlet:ram_recovery', 0.9980),
des_vars.add_output('fan:PRdes', 1.300),
des_vars.add_output('fan:effDes', 0.96888),
des_vars.add_output('fan:effPoly', 0.97),
des_vars.add_output('splitter:BPR', 23.7281), #23.9878
des_vars.add_output('duct2:dPqP', 0.0100),
des_vars.add_output('lpc:PRdes', 3.000),
des_vars.add_output('lpc:effDes', 0.889513),
des_vars.add_output('lpc:effPoly', 0.905),
des_vars.add_output('duct25:dPqP', 0.0150),
des_vars.add_output('hpc:PRdes', 14.103),
des_vars.add_output('OPR', 53.6332) #53.635)
des_vars.add_output('OPR_simple', 55.0)
des_vars.add_output('hpc:effDes', 0.847001),
des_vars.add_output('hpc:effPoly', 0.89),
des_vars.add_output('burner:dPqP', 0.0400),
des_vars.add_output('hpt:effDes', 0.922649),
des_vars.add_output('hpt:effPoly', 0.91),
des_vars.add_output('duct45:dPqP', 0.0050),
des_vars.add_output('lpt:effDes', 0.940104),
des_vars.add_output('lpt:effPoly', 0.92),
des_vars.add_output('duct5:dPqP', 0.0100),
des_vars.add_output('core_nozz:Cv', 0.9999),
des_vars.add_output('duct17:dPqP', 0.0150),
des_vars.add_output('byp_nozz:Cv', 0.9975),
des_vars.add_output('fan_shaft:Nmech', 2184.5, units='rpm'),
des_vars.add_output('lp_shaft:Nmech', 6772.0, units='rpm'),
des_vars.add_output('lp_shaft:fracLoss', 0.01)
des_vars.add_output('hp_shaft:Nmech', 20871.0, units='rpm'),
des_vars.add_output('hp_shaft:HPX', 350.0, units='hp'),

des_vars.add_output('bld25:sbv:frac_W', 0.0),
des_vars.add_output('hpc:bld_inlet:frac_W', 0.0),
des_vars.add_output('hpc:bld_inlet:frac_P', 0.1465),
des_vars.add_output('hpc:bld_inlet:frac_work', 0.5),
des_vars.add_output('hpc:bld_exit:frac_W', 0.02),
des_vars.add_output('hpc:bld_exit:frac_P', 0.1465),
des_vars.add_output('hpc:bld_exit:frac_work', 0.5),
des_vars.add_output('hpc:cust:frac_W', 0.0),
des_vars.add_output('hpc:cust:frac_P', 0.1465),
des_vars.add_output('hpc:cust:frac_work', 0.35),
des_vars.add_output('bld3:bld_inlet:frac_W', 0.063660111), #different than NPSS due to Wref
des_vars.add_output('bld3:bld_exit:frac_W', 0.07037185), #different than NPSS due to Wref
des_vars.add_output('hpt:bld_inlet:frac_P', 1.0),
des_vars.add_output('hpt:bld_exit:frac_P', 0.0),
des_vars.add_output('lpt:bld_inlet:frac_P', 1.0),
des_vars.add_output('lpt:bld_exit:frac_P', 0.0),
des_vars.add_output('bypBld:frac_W', 0.0),

des_vars.add_output('inlet:MN_out', 0.625),
des_vars.add_output('fan:MN_out', 0.45)
des_vars.add_output('splitter:MN_out1', 0.45)
des_vars.add_output('splitter:MN_out2', 0.45)
des_vars.add_output('duct2:MN_out', 0.45),
des_vars.add_output('lpc:MN_out', 0.45),
des_vars.add_output('bld25:MN_out', 0.45),
des_vars.add_output('duct25:MN_out', 0.45),
des_vars.add_output('hpc:MN_out', 0.30),
des_vars.add_output('bld3:MN_out', 0.30)
des_vars.add_output('burner:MN_out', 0.10),
des_vars.add_output('hpt:MN_out', 0.30),
des_vars.add_output('duct45:MN_out', 0.45),
des_vars.add_output('lpt:MN_out', 0.35),
des_vars.add_output('duct5:MN_out', 0.25),
des_vars.add_output('bypBld:MN_out', 0.45),
des_vars.add_output('duct17:MN_out', 0.45),

# POINT 1: Top-of-climb (TOC)
des_vars.add_output('TOC:alt', 35000., units='ft'),
des_vars.add_output('TOC:MN', 0.8),
des_vars.add_output('TOC:T4max', 3150.0, units='degR'),
# des_vars.add_output('FAR', 0.02833)
des_vars.add_output('TOC:Fn_des', 6073.4, units='lbf'),
des_vars.add_output('TOC:W', 820.951, units='lbm/s')

# POINT 2: Rolling Takeoff (RTO)
des_vars.add_output('RTO:MN', 0.25),
des_vars.add_output('RTO:alt', 0.0, units='ft'),
des_vars.add_output('RTO:Fn_target', 22800.0, units='lbf'), #8950.0
des_vars.add_output('RTO:dTs', 27.0, units='degR')
des_vars.add_output('RTO:Ath', 5532.3, units='inch**2')
des_vars.add_output('RTO:RlineMap', 1.75)
des_vars.add_output('RTO:T4max', 3400.0, units='degR')
des_vars.add_output('RTO:W', 1916.13, units='lbm/s')
des_vars.add_output('RTO:BPR', 25.5620)
des_vars.add_output('RTO:ram_recovery', 0.9970)

# POINT 3: Sea-Level Static (SLS)
des_vars.add_output('SLS:MN', 0.000001),
des_vars.add_output('SLS:alt', 0.0, units='ft'),
des_vars.add_output('SLS:Fn_target', 28620.9, units='lbf'), #8950.0
des_vars.add_output('SLS:dTs', 27.0, units='degR')
des_vars.add_output('SLS:Ath', 6315.6, units='inch**2')
des_vars.add_output('SLS:RlineMap', 1.75)

# POINT 4: Cruise (CRZ)
des_vars.add_output('CRZ:MN', 0.8),
des_vars.add_output('CRZ:alt', 35000.0, units='ft'),
des_vars.add_output('CRZ:Fn_target', 5466.5, units='lbf'), #8950.0
des_vars.add_output('CRZ:dTs', 0.0, units='degR')
des_vars.add_output('CRZ:Ath', 4747.1, units='inch**2')
des_vars.add_output('CRZ:RlineMap', 1.9401)


# TOC POINT (DESIGN)
prob.model.add_subsystem('TOC', N3(statics=True))

prob.model.connect('TOC:alt', 'TOC.fc.alt')
prob.model.connect('TOC:MN', 'TOC.fc.MN')
# prob.model.connect('TOC:Fn_des', 'TOC.balance.rhs:W')
prob.model.connect('TOC:T4max', 'TOC.balance.rhs:FAR')
# prob.model.connect('FAR','TOC.burner.Fl_I:FAR')
prob.model.connect('TOC:W', 'TOC.fc.W')

prob.model.connect('inlet:ram_recovery', 'TOC.inlet.ram_recovery')
prob.model.connect('fan:PRdes', ['TOC.fan.PR', 'TOC.opr_calc.FPR'])
# prob.model.connect('fan:effDes', 'TOC.fan.map.effDes')
prob.model.connect('fan:effPoly', 'TOC.balance.rhs:fan_eff')
# prob.model.connect('splitter:BPR', 'TOC.splitter.BPR')
prob.model.connect('duct2:dPqP', 'TOC.duct2.dPqP')
prob.model.connect('lpc:PRdes', ['TOC.lpc.PR', 'TOC.opr_calc.LPCPR'])
# prob.model.connect('lpc:effDes', 'TOC.lpc.map.effDes')
prob.model.connect('lpc:effPoly', 'TOC.balance.rhs:lpc_eff')
prob.model.connect('duct25:dPqP', 'TOC.duct25.dPqP')
# prob.model.connect('hpc:PRdes', 'TOC.hpc.PR')
# prob.model.connect('OPR', 'TOC.balance.rhs:hpc_PR')
prob.model.connect('OPR_simple', 'TOC.balance.rhs:hpc_PR')
# prob.model.connect('hpc:effDes', 'TOC.hpc.map.effDes')
# prob.model.connect('hpc:effPoly', 'TOC.balance.rhs:hpc_eff')
prob.model.connect('burner:dPqP', 'TOC.burner.dPqP')
# prob.model.connect('hpt:effDes', 'TOC.hpt.map.effDes')
prob.model.connect('hpt:effPoly', 'TOC.balance.rhs:hpt_eff')
prob.model.connect('duct45:dPqP', 'TOC.duct45.dPqP')
# prob.model.connect('lpt:effDes', 'TOC.lpt.map.effDes')
prob.model.connect('lpt:effPoly', 'TOC.balance.rhs:lpt_eff')
prob.model.connect('duct5:dPqP', 'TOC.duct5.dPqP')
prob.model.connect('core_nozz:Cv', ['TOC.core_nozz.Cv', 'TOC.ext_ratio.core_Cv'])
prob.model.connect('duct17:dPqP', 'TOC.duct17.dPqP')
prob.model.connect('byp_nozz:Cv', ['TOC.byp_nozz.Cv', 'TOC.ext_ratio.byp_Cv'])
prob.model.connect('fan_shaft:Nmech', 'TOC.Fan_Nmech')
prob.model.connect('lp_shaft:Nmech', 'TOC.LP_Nmech')
prob.model.connect('lp_shaft:fracLoss', 'TOC.lp_shaft.fracLoss')
prob.model.connect('hp_shaft:Nmech', 'TOC.HP_Nmech')
prob.model.connect('hp_shaft:HPX', 'TOC.hp_shaft.HPX')

prob.model.connect('bld25:sbv:frac_W', 'TOC.bld25.sbv:frac_W')
prob.model.connect('hpc:bld_inlet:frac_W', 'TOC.hpc.bld_inlet:frac_W')
prob.model.connect('hpc:bld_inlet:frac_P', 'TOC.hpc.bld_inlet:frac_P')
prob.model.connect('hpc:bld_inlet:frac_work', 'TOC.hpc.bld_inlet:frac_work')
prob.model.connect('hpc:bld_exit:frac_W', 'TOC.hpc.bld_exit:frac_W')
prob.model.connect('hpc:bld_exit:frac_P', 'TOC.hpc.bld_exit:frac_P')
prob.model.connect('hpc:bld_exit:frac_work', 'TOC.hpc.bld_exit:frac_work')
prob.model.connect('bld3:bld_inlet:frac_W', 'TOC.bld3.bld_inlet:frac_W')
prob.model.connect('bld3:bld_exit:frac_W', 'TOC.bld3.bld_exit:frac_W')
prob.model.connect('hpc:cust:frac_W', 'TOC.hpc.cust:frac_W')
prob.model.connect('hpc:cust:frac_P', 'TOC.hpc.cust:frac_P')
prob.model.connect('hpc:cust:frac_work', 'TOC.hpc.cust:frac_work')
prob.model.connect('hpt:bld_inlet:frac_P', 'TOC.hpt.bld_inlet:frac_P')
prob.model.connect('hpt:bld_exit:frac_P', 'TOC.hpt.bld_exit:frac_P')
prob.model.connect('lpt:bld_inlet:frac_P', 'TOC.lpt.bld_inlet:frac_P')
prob.model.connect('lpt:bld_exit:frac_P', 'TOC.lpt.bld_exit:frac_P')
prob.model.connect('bypBld:frac_W', 'TOC.byp_bld.bypBld:frac_W')

prob.model.connect('inlet:MN_out', 'TOC.inlet.MN')
prob.model.connect('fan:MN_out', 'TOC.fan.MN')
prob.model.connect('splitter:MN_out1', 'TOC.splitter.MN1')
prob.model.connect('splitter:MN_out2', 'TOC.splitter.MN2')
prob.model.connect('duct2:MN_out', 'TOC.duct2.MN')
prob.model.connect('lpc:MN_out', 'TOC.lpc.MN')
prob.model.connect('bld25:MN_out', 'TOC.bld25.MN')
prob.model.connect('duct25:MN_out', 'TOC.duct25.MN')
prob.model.connect('hpc:MN_out', 'TOC.hpc.MN')
prob.model.connect('bld3:MN_out', 'TOC.bld3.MN')
prob.model.connect('burner:MN_out', 'TOC.burner.MN')
prob.model.connect('hpt:MN_out', 'TOC.hpt.MN')
prob.model.connect('duct45:MN_out', 'TOC.duct45.MN')
prob.model.connect('lpt:MN_out', 'TOC.lpt.MN')
prob.model.connect('duct5:MN_out', 'TOC.duct5.MN')
prob.model.connect('bypBld:MN_out', 'TOC.byp_bld.MN')
prob.model.connect('duct17:MN_out', 'TOC.duct17.MN')





# OTHER POINTS (OFF-DESIGN)
pts = []
# pts = ['RTO','SLS','CRZ']
OD_statics = True


# prob.model.connect('RTO:Fn_target', 'RTO.balance.rhs:FAR')
# prob.model.connect('SLS:Fn_target', 'SLS.balance.rhs:FAR')
# prob.model.connect('CRZ:Fn_target', 'CRZ.balance.rhs:FAR')

# prob.model.add_subsystem('RTO', N3(design=False, statics=OD_statics, cooling=True))
# prob.model.add_subsystem('RTO', N3(design=False, statics=OD_statics))
# prob.model.add_subsystem('SLS', N3(design=False, statics=OD_statics))
# prob.model.add_subsystem('CRZ', N3(design=False, statics=OD_statics))


for pt in pts:
    # ODpt.nonlinear_solver.options['maxiter'] = 0

    prob.model.connect(pt+':alt', pt+'.fc.alt')
    prob.model.connect(pt+':MN', pt+'.fc.MN')
    prob.model.connect(pt+':Fn_target', pt+'.balance.rhs:FAR')
    prob.model.connect(pt+':dTs', pt+'.fc.dTs')
    # prob.model.connect(pt+':Ath',pt+'.balance.rhs:BPR')
    prob.model.connect(pt+':RlineMap',pt+'.balance.rhs:BPR')
    # prob.model.connect(pt+':T4max', pt+'.balance.rhs:FAR')

    # prob.model.connect(pt+':cust_fracW', pt+'.hpc.cust:frac_W')

    prob.model.connect('RTO:ram_recovery', pt+'.inlet.ram_recovery')
    # prob.model.connect('RTO:BPR', pt+'.splitter.BPR')
    prob.model.connect('duct2:dPqP', pt+'.duct2.dPqP')
    prob.model.connect('duct25:dPqP', pt+'.duct25.dPqP')
    prob.model.connect('burner:dPqP', pt+'.burner.dPqP')
    prob.model.connect('duct45:dPqP', pt+'.duct45.dPqP')
    prob.model.connect('duct5:dPqP', pt+'.duct5.dPqP')
    prob.model.connect('core_nozz:Cv', pt+'.core_nozz.Cv')
    prob.model.connect('duct17:dPqP', pt+'.duct17.dPqP')
    prob.model.connect('byp_nozz:Cv', pt+'.byp_nozz.Cv')
    prob.model.connect('lp_shaft:fracLoss', pt+'.lp_shaft.fracLoss')
    prob.model.connect('hp_shaft:HPX', pt+'.hp_shaft.HPX')

    prob.model.connect('bld25:sbv:frac_W', pt+'.bld25.sbv:frac_W')
    prob.model.connect('hpc:bld_inlet:frac_W', pt+'.hpc.bld_inlet:frac_W')
    prob.model.connect('hpc:bld_inlet:frac_P', pt+'.hpc.bld_inlet:frac_P')
    prob.model.connect('hpc:bld_inlet:frac_work', pt+'.hpc.bld_inlet:frac_work')
    prob.model.connect('hpc:bld_exit:frac_W', pt+'.hpc.bld_exit:frac_W')
    prob.model.connect('hpc:bld_exit:frac_P', pt+'.hpc.bld_exit:frac_P')
    prob.model.connect('hpc:bld_exit:frac_work', pt+'.hpc.bld_exit:frac_work')
    prob.model.connect('bld3:bld_inlet:frac_W', pt+'.bld3.bld_inlet:frac_W')
    prob.model.connect('bld3:bld_exit:frac_W', pt+'.bld3.bld_exit:frac_W')
    # prob.model.connect('TOC.balance.hpt_chrg_cool_frac', pt+'.bld3.bld_inlet:frac_W')
    # prob.model.connect('TOC.balance.hpt_nochrg_cool_frac', pt+'.bld3.bld_exit:frac_W')
    prob.model.connect('hpc:cust:frac_W', pt+'.hpc.cust:frac_W')
    prob.model.connect('hpc:cust:frac_P', pt+'.hpc.cust:frac_P')
    prob.model.connect('hpc:cust:frac_work', pt+'.hpc.cust:frac_work')
    prob.model.connect('hpt:bld_inlet:frac_P', pt+'.hpt.bld_inlet:frac_P')
    prob.model.connect('hpt:bld_exit:frac_P', pt+'.hpt.bld_exit:frac_P')
    prob.model.connect('lpt:bld_inlet:frac_P', pt+'.lpt.bld_inlet:frac_P')
    prob.model.connect('lpt:bld_exit:frac_P', pt+'.lpt.bld_exit:frac_P')
    prob.model.connect('bypBld:frac_W', pt+'.byp_bld.bypBld:frac_W')

    prob.model.connect('TOC.fan.s_PR', pt+'.fan.s_PR')
    prob.model.connect('TOC.fan.s_Wc', pt+'.fan.s_Wc')
    prob.model.connect('TOC.fan.s_eff', pt+'.fan.s_eff')
    prob.model.connect('TOC.fan.s_Nc', pt+'.fan.s_Nc')
    prob.model.connect('TOC.lpc.s_PR', pt+'.lpc.s_PR')
    prob.model.connect('TOC.lpc.s_Wc', pt+'.lpc.s_Wc')
    prob.model.connect('TOC.lpc.s_eff', pt+'.lpc.s_eff')
    prob.model.connect('TOC.lpc.s_Nc', pt+'.lpc.s_Nc')
    prob.model.connect('TOC.hpc.s_PR', pt+'.hpc.s_PR')
    prob.model.connect('TOC.hpc.s_Wc', pt+'.hpc.s_Wc')
    prob.model.connect('TOC.hpc.s_eff', pt+'.hpc.s_eff')
    prob.model.connect('TOC.hpc.s_Nc', pt+'.hpc.s_Nc')
    prob.model.connect('TOC.hpt.s_PR', pt+'.hpt.s_PR')
    prob.model.connect('TOC.hpt.s_Wp', pt+'.hpt.s_Wp')
    prob.model.connect('TOC.hpt.s_eff', pt+'.hpt.s_eff')
    prob.model.connect('TOC.hpt.s_Np', pt+'.hpt.s_Np')
    prob.model.connect('TOC.lpt.s_PR', pt+'.lpt.s_PR')
    prob.model.connect('TOC.lpt.s_Wp', pt+'.lpt.s_Wp')
    prob.model.connect('TOC.lpt.s_eff', pt+'.lpt.s_eff')
    prob.model.connect('TOC.lpt.s_Np', pt+'.lpt.s_Np')

    prob.model.connect('TOC.gearbox.gear_ratio', pt+'.gearbox.gear_ratio')
    # prob.model.connect('TOC.byp_nozz.Throat:stat:area',pt+'.balance.rhs:BPR')

    prob.model.connect('TOC.core_nozz.Throat:stat:area',pt+'.balance.rhs:W')
    # prob.model.connect('RTO:W', pt+'.fc.W')

    if OD_statics:
        prob.model.connect('TOC.inlet.Fl_O:stat:area', pt+'.inlet.area')
        prob.model.connect('TOC.fan.Fl_O:stat:area', pt+'.fan.area')
        prob.model.connect('TOC.splitter.Fl_O1:stat:area', pt+'.splitter.area1')
        prob.model.connect('TOC.splitter.Fl_O2:stat:area', pt+'.splitter.area2')
        prob.model.connect('TOC.duct2.Fl_O:stat:area', pt+'.duct2.area')
        prob.model.connect('TOC.lpc.Fl_O:stat:area', pt+'.lpc.area')
        prob.model.connect('TOC.bld25.Fl_O:stat:area', pt+'.bld25.area')
        prob.model.connect('TOC.duct25.Fl_O:stat:area', pt+'.duct25.area')
        prob.model.connect('TOC.hpc.Fl_O:stat:area', pt+'.hpc.area')
        prob.model.connect('TOC.bld3.Fl_O:stat:area', pt+'.bld3.area')
        prob.model.connect('TOC.burner.Fl_O:stat:area', pt+'.burner.area')
        prob.model.connect('TOC.hpt.Fl_O:stat:area', pt+'.hpt.area')
        prob.model.connect('TOC.duct45.Fl_O:stat:area', pt+'.duct45.area')
        prob.model.connect('TOC.lpt.Fl_O:stat:area', pt+'.lpt.area')
        prob.model.connect('TOC.duct5.Fl_O:stat:area', pt+'.duct5.area')
        prob.model.connect('TOC.byp_bld.Fl_O:stat:area', pt+'.byp_bld.area')
        prob.model.connect('TOC.duct17.Fl_O:stat:area', pt+'.duct17.area')


# prob.model.connect('RTO.balance.hpt_chrg_cool_frac', 'TOC.bld3.bld_inlet:frac_W')
# prob.model.connect('RTO.balance.hpt_nochrg_cool_frac', 'TOC.bld3.bld_exit:frac_W')

# prob.model.connect('RTO.balance.hpt_chrg_cool_frac', 'SLS.bld3.bld_inlet:frac_W')
# prob.model.connect('RTO.balance.hpt_nochrg_cool_frac', 'SLS.bld3.bld_exit:frac_W')

# prob.model.connect('RTO.balance.hpt_chrg_cool_frac', 'CRZ.bld3.bld_inlet:frac_W')
# prob.model.connect('RTO.balance.hpt_nochrg_cool_frac', 'CRZ.bld3.bld_exit:frac_W')

# bal = prob.model.add_subsystem('bal', BalanceComp())
# bal.add_balance('TOC_BPR', val=25.0, units=None, mult_val=1.4, eq_units='ft/s', use_mult=True)
# prob.model.connect('bal.TOC_BPR', 'TOC.splitter.BPR')
# prob.model.connect('CRZ.byp_nozz.Fl_O:stat:V', 'bal.lhs:TOC_BPR')
# prob.model.connect('CRZ.core_nozz.Fl_O:stat:V', 'bal.rhs:TOC_BPR')

# bal.add_balance('TOC_W', val=800.0, units='lbm/s', eq_units='degR')
# prob.model.connect('bal.TOC_W', 'TOC.inlet.Fl_I:stat:W')
# prob.model.connect('RTO.burner.Fl_O:tot:T', 'bal.lhs:TOC_W')
# prob.model.connect('RTO:T4max','bal.rhs:TOC_W')

# bal.add_balance('CRZ_Fn_target', val=6000.0, units='lbf', eq_units='lbf', use_mult=True, mult_val=0.9)
# prob.model.connect('bal.CRZ_Fn_target', 'CRZ.balance.rhs:FAR')
# prob.model.connect('TOC.perf.Fn', 'bal.lhs:CRZ_Fn_target')
# prob.model.connect('CRZ.perf.Fn','bal.rhs:CRZ_Fn_target')

# bal.add_balance('SLS_Fn_target', val=28000.0, units='lbf', eq_units='lbf', use_mult=True, mult_val=1.2553)
# prob.model.connect('bal.SLS_Fn_target', 'SLS.balance.rhs:FAR')
# prob.model.connect('RTO.perf.Fn', 'bal.lhs:SLS_Fn_target')
# prob.model.connect('SLS.perf.Fn','bal.rhs:SLS_Fn_target')

# newton = prob.model.nonlinear_solver = NewtonSolver()
# newton.options['atol'] = 1e-6
# newton.options['rtol'] = 1e-6
# newton.options['iprint'] = 2
# newton.options['maxiter'] = 20
# newton.options['solve_subsystems'] = True
# newton.options['max_sub_solves'] = 100
# # newton.linesearch =  ArmijoGoldsteinLS()
# newton.linesearch =  BoundsEnforceLS()
# newton.linesearch.options['maxiter'] = 2
# newton.linesearch.options['bound_enforcement'] = 'scalar'
# newton.linesearch.options['iprint'] = -1
# newton.linesearch.options['print_bound_enforce'] = False
# # newton.linesearch.options['alpha'] = 0.5

# prob.model.linear_solver = DirectSolver()
# prob.model.jacobian = CSCJacobian()

# prob.model.linear_solver = ScipyKrylov()
# prob.model.linear_solver.options['iprint'] = 2
# prob.model.linear_solver.precon = DirectSolver()
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


recorder = SqliteRecorder('N3_SPD.sql')

prob.model.add_recorder(recorder)
prob.model.recording_options['record_outputs'] = True
prob.model.recording_options['record_inputs'] = True
# prob.model.recording_options['record_responses'] = True
# prob.model.recording_options['record_objectives'] = True
# prob.model.recording_options['record_constraints'] = True

prob.setup(check=False)

# prob['RTO.hpt_cooling.x_factor'] = 0.9

# initial guesses
prob['TOC.balance.FAR'] = 0.02833
# prob['bal.TOC_W'] = 813.5
prob['TOC.balance.lpt_PR'] = 11.078
prob['TOC.balance.hpt_PR'] = 4.115
prob['TOC.fc.balance.Pt'] = 5.2
prob['TOC.fc.balance.Tt'] = 440.0

for pt in pts:

    if pt == 'RTO':
        prob[pt+'.balance.FAR'] = 0.02999
        prob[pt+'.balance.W'] = 1903.84
        prob[pt+'.balance.BPR'] = 25.7412
        prob[pt+'.balance.fan_Nmech'] = 2140.0
        prob[pt+'.balance.lp_Nmech'] = 6634.0
        prob[pt+'.balance.hp_Nmech'] = 22269.0
        prob[pt+'.fc.balance.Pt'] = 15.349
        prob[pt+'.fc.balance.Tt'] = 552.49
        prob[pt+'.hpt.PR'] = 4.138
        prob[pt+'.lpt.PR'] = 8.322
        prob[pt+'.fan.map.RlineMap'] = 1.7500
        prob[pt+'.lpc.map.RlineMap'] = 2.0111
        prob[pt+'.hpc.map.RlineMap'] = 2.0659
        prob[pt+'.gearbox.trq_base'] = 52407.8

    if pt == 'SLS':
        prob[pt+'.balance.FAR'] = 0.02682
        prob[pt+'.balance.W'] = 1723.88
        prob[pt+'.balance.BPR'] = 27.4820
        prob[pt+'.balance.fan_Nmech'] = 1960.8
        prob[pt+'.balance.lp_Nmech'] = 6078.6
        prob[pt+'.balance.hp_Nmech'] = 21582.7
        prob[pt+'.fc.balance.Pt'] = 14.696
        prob[pt+'.fc.balance.Tt'] = 545.67
        prob[pt+'.hpt.PR'] = 4.173
        prob[pt+'.lpt.PR'] = 7.155
        prob[pt+'.fan.map.RlineMap'] = 1.7500
        prob[pt+'.lpc.map.RlineMap'] = 1.8738
        prob[pt+'.hpc.map.RlineMap'] = 2.0323
        prob[pt+'.gearbox.trq_base'] = 41806.8

    if pt == 'CRZ':
        prob[pt+'.balance.FAR'] = 0.02638
        prob[pt+'.balance.W'] = 795.72
        prob[pt+'.balance.BPR'] = 24.5257
        prob[pt+'.balance.fan_Nmech'] = 2119.1
        prob[pt+'.balance.lp_Nmech'] = 6569.3
        prob[pt+'.balance.hp_Nmech'] = 20511.8
        prob[pt+'.fc.balance.Pt'] = 5.272
        prob[pt+'.fc.balance.Tt'] = 444.41
        prob[pt+'.hpt.PR'] = 4.126
        prob[pt+'.lpt.PR'] = 10.954
        prob[pt+'.fan.map.RlineMap'] = 1.9401
        prob[pt+'.lpc.map.RlineMap'] = 2.1099
        prob[pt+'.hpc.map.RlineMap'] = 1.9772
        prob[pt+'.gearbox.trq_base'] = 22179.6



st = time.time()

# prob.model.RTO.nonlinear_solver.options['maxiter']=1
# prob.model.nonlinear_solver.linesearch.options['print_bound_enforce'] = True

# from openmdao.api import view_model
# view_model(prob)
# exit()

prob.set_solver_print(level=-1)
prob.set_solver_print(level=2, depth=1)
prob.run_model()
# prob.check_partials(comps=['OD1.gearbox'], compact_print=True)

# prob.model.list_outputs(residuals=True)

# prob.check_partials(compact_print=False,abs_err_tol=1e-3, rel_err_tol=1e-3)
# exit()

for pt in ['TOC']+pts:
    viewer(prob, pt)


print()
print("time", time.time() - st)

prob.model.list_outputs(explicit=False, residuals=True, residuals_tol=1e-6)
