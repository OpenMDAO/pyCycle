import time

import openmdao.api as om 

from simple_turboshaft import Turboshaft, viewer

prob = om.Problem()

des_vars = prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=["*"])

# Design point inputs
des_vars.add_output('alt', 0.0, units='ft'),
des_vars.add_output('MN', 0.000001),
des_vars.add_output('T4max', 2370.0, units='degR'),
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

prob.setup(check=False)

# Set initial guesses for balances
prob['DESIGN.balance.FAR'] = 0.0175506829934
prob['DESIGN.balance.W'] = 27.265
prob['DESIGN.balance.turb_PR'] = 3.8768
prob['DESIGN.balance.pt_PR'] = 2.8148
prob['DESIGN.fc.balance.Pt'] = 14.6955113159
prob['DESIGN.fc.balance.Tt'] = 518.665288153

st = time.time()

prob.set_solver_print(level=-1)
prob.set_solver_print(level=2, depth=1)
prob.run_model()

viewer(prob, 'DESIGN')

print()
print("time", time.time() - st)





