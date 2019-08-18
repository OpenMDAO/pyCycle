import time
import openmdao.api as om

from simple_turboshaft import Turboshaft, viewer

prob = om.Problem()

des_vars = prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=["*"])

# Design point inputs
des_vars.add_output('alt', 0.0, units='ft'),
des_vars.add_output('MN', 0.000001),
des_vars.add_output('T4max', 2370.0, units='degR'),
# des_vars.add_output('Fn_des', 11800.0, units='lbf'),
des_vars.add_output('pwr_des', 4000.0, units='hp')
des_vars.add_output('nozz_PR', 1.2)
des_vars.add_output('comp:PRdes', 13.5),
des_vars.add_output('comp:effDes', 0.83),
des_vars.add_output('burn:dPqP', 0.03),
des_vars.add_output('turb:effDes', 0.86),
des_vars.add_output('pt:effDes', 0.9),    
des_vars.add_output('nozz:Cv', 0.99),
des_vars.add_output('HP_shaft:Nmech', 8070.0, units='rpm'),
des_vars.add_output('LP_shaft:Nmech', 5000.0, units='rpm'),
des_vars.add_output('inlet:MN_out', 0.60),
des_vars.add_output('comp:MN_out', 0.20),
des_vars.add_output('burner:MN_out', 0.20),
des_vars.add_output('turb:MN_out', 0.4),
des_vars.add_output('pt:MN_out', 0.5),

# Off-design (point 1) inputs
des_vars.add_output('OD1_MN', 0.000001),
des_vars.add_output('OD1_alt', 0.0, units='ft'),
des_vars.add_output('OD1_pwr', 3500.0, units='hp')
des_vars.add_output('OD1_LP_Nmech', 5000., units='rpm')

# Create design instance of model
prob.model.add_subsystem('DESIGN', Turboshaft())

# Connect design point inputs to model
prob.model.connect('alt', 'DESIGN.fc.alt')
prob.model.connect('MN', 'DESIGN.fc.MN')
# prob.model.connect('Fn_des', 'DESIGN.balance.rhs:W')
prob.model.connect('T4max', 'DESIGN.balance.rhs:FAR')
prob.model.connect('pwr_des', 'DESIGN.balance.rhs:pt_PR')
prob.model.connect('nozz_PR', 'DESIGN.balance.rhs:W')

prob.model.connect('comp:PRdes', 'DESIGN.comp.PR')
prob.model.connect('comp:effDes', 'DESIGN.comp.eff')
prob.model.connect('burn:dPqP', 'DESIGN.burner.dPqP')
prob.model.connect('turb:effDes', 'DESIGN.turb.eff')
prob.model.connect('pt:effDes', 'DESIGN.pt.eff')
prob.model.connect('nozz:Cv', 'DESIGN.nozz.Cv')
prob.model.connect('HP_shaft:Nmech', 'DESIGN.HP_Nmech')
prob.model.connect('LP_shaft:Nmech', 'DESIGN.LP_Nmech')

prob.model.connect('inlet:MN_out', 'DESIGN.inlet.MN')
prob.model.connect('comp:MN_out', 'DESIGN.comp.MN')
prob.model.connect('burner:MN_out', 'DESIGN.burner.MN')
prob.model.connect('turb:MN_out', 'DESIGN.turb.MN')

# Connect off-design and required design inputs to model
pts = ['OD1']

for pt in pts:
    prob.model.add_subsystem(pt, Turboshaft(design=False))

    prob.model.connect('burn:dPqP', pt+'.burner.dPqP')
    prob.model.connect('nozz:Cv', pt+'.nozz.Cv')

    prob.model.connect(pt+'_alt', pt+'.fc.alt')
    prob.model.connect(pt+'_MN', pt+'.fc.MN')
    prob.model.connect(pt+'_LP_Nmech', pt+'.LP_Nmech')
    prob.model.connect(pt+'_pwr', pt+'.balance.rhs:FAR')

    prob.model.connect('DESIGN.comp.s_PR', pt+'.comp.s_PR')
    prob.model.connect('DESIGN.comp.s_Wc', pt+'.comp.s_Wc')
    prob.model.connect('DESIGN.comp.s_eff', pt+'.comp.s_eff')
    prob.model.connect('DESIGN.comp.s_Nc', pt+'.comp.s_Nc')

    prob.model.connect('DESIGN.turb.s_PR', pt+'.turb.s_PR')
    prob.model.connect('DESIGN.turb.s_Wp', pt+'.turb.s_Wp')
    prob.model.connect('DESIGN.turb.s_eff', pt+'.turb.s_eff')
    prob.model.connect('DESIGN.turb.s_Np', pt+'.turb.s_Np')

    prob.model.connect('DESIGN.pt.s_PR', pt+'.pt.s_PR')
    prob.model.connect('DESIGN.pt.s_Wp', pt+'.pt.s_Wp')
    prob.model.connect('DESIGN.pt.s_eff', pt+'.pt.s_eff')
    prob.model.connect('DESIGN.pt.s_Np', pt+'.pt.s_Np')

    prob.model.connect('DESIGN.inlet.Fl_O:stat:area', pt+'.inlet.area')
    prob.model.connect('DESIGN.comp.Fl_O:stat:area', pt+'.comp.area')
    prob.model.connect('DESIGN.burner.Fl_O:stat:area', pt+'.burner.area')
    prob.model.connect('DESIGN.turb.Fl_O:stat:area', pt+'.turb.area')
    prob.model.connect('DESIGN.pt.Fl_O:stat:area', pt+'.pt.area')

    # prob.model.connect(pt+'_T4', pt+'.balance.rhs:FAR')
    # prob.model.connect(pt+'_pwr', pt+'.balance.rhs:FAR')
    prob.model.connect('DESIGN.nozz.Throat:stat:area', pt+'.balance.rhs:W')


prob.setup(check=False)

# Set initial guesses for balances
prob['DESIGN.balance.FAR'] = 0.0175506829934
prob['DESIGN.balance.W'] = 27.265
prob['DESIGN.balance.turb_PR'] = 3.8768
prob['DESIGN.balance.pt_PR'] = 2.8148
prob['DESIGN.fc.balance.Pt'] = 14.6955113159
prob['DESIGN.fc.balance.Tt'] = 518.665288153

for pt in pts:
    prob[pt+'.balance.W'] = 27.265
    prob[pt+'.balance.FAR'] = 0.0175506829934
    prob[pt+'.balance.HP_Nmech'] = 8070.0
    prob[pt+'.fc.balance.Pt'] = 15.703
    prob[pt+'.fc.balance.Tt'] = 558.31
    prob[pt+'.turb.PR'] = 3.8768
    prob[pt+'.pt.PR'] = 2.8148

st = time.time()

prob.set_solver_print(level=-1)
prob.set_solver_print(level=2, depth=1)
prob.run_model()

for pt in ['DESIGN']+pts:
    viewer(prob, pt)


print()
print("time", time.time() - st)





