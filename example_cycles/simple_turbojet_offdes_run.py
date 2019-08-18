import time
from openmdao.api import Problem, IndepVarComp
from openmdao.utils.units import convert_units as cu

from simple_turbojet import Turbojet, viewer

prob = Problem()

des_vars = prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=["*"])

# Design point inputs
des_vars.add_output('alt', 0.0, units='ft'),
des_vars.add_output('MN', 0.000001),
des_vars.add_output('T4max', 2370.0, units='degR'),
des_vars.add_output('Fn_des', 11800.0, units='lbf'),
des_vars.add_output('comp:PRdes', 13.5),
des_vars.add_output('comp:effDes', 0.83),
des_vars.add_output('burn:dPqP', 0.03),
des_vars.add_output('turb:effDes', 0.86),
des_vars.add_output('nozz:Cv', 0.99),
des_vars.add_output('shaft:Nmech', 8070.0, units='rpm'),
des_vars.add_output('inlet:MN_out', 0.60),
des_vars.add_output('comp:MN_out', 0.20),
des_vars.add_output('burner:MN_out', 0.20),
des_vars.add_output('turb:MN_out', 0.4),

# Off-design (point 1) inputs
des_vars.add_output('OD1_MN', 0.000001),
des_vars.add_output('OD1_alt', 0.0, units='ft'),
des_vars.add_output('OD1_Fn', 11000.0, units='lbf')

# Create design instance of model
prob.model.add_subsystem('DESIGN', Turbojet())

# Connect design point inputs to model
prob.model.connect('alt', 'DESIGN.fc.alt')
prob.model.connect('MN', 'DESIGN.fc.MN')
prob.model.connect('Fn_des', 'DESIGN.balance.rhs:W')
prob.model.connect('T4max', 'DESIGN.balance.rhs:FAR')

prob.model.connect('comp:PRdes', 'DESIGN.comp.PR')
prob.model.connect('comp:effDes', 'DESIGN.comp.eff')
prob.model.connect('burn:dPqP', 'DESIGN.burner.dPqP')
prob.model.connect('turb:effDes', 'DESIGN.turb.eff')
prob.model.connect('nozz:Cv', 'DESIGN.nozz.Cv')
prob.model.connect('shaft:Nmech', 'DESIGN.Nmech')

prob.model.connect('inlet:MN_out', 'DESIGN.inlet.MN')
prob.model.connect('comp:MN_out', 'DESIGN.comp.MN')
prob.model.connect('burner:MN_out', 'DESIGN.burner.MN')
prob.model.connect('turb:MN_out', 'DESIGN.turb.MN')

# Connect off-design and required design inputs to model
pts = ['OD1']
for pt in pts:
    prob.model.add_subsystem(pt, Turbojet(design=False))

    prob.model.connect(pt+'_alt', pt+'.fc.alt')
    prob.model.connect(pt+'_MN', pt+'.fc.MN')
    prob.model.connect(pt+'_Fn', pt+'.balance.rhs:FAR')

    prob.model.connect('burn:dPqP', pt+'.burner.dPqP')
    prob.model.connect('nozz:Cv', pt+'.nozz.Cv')

    prob.model.connect('DESIGN.comp.s_PR', pt+'.comp.s_PR')
    prob.model.connect('DESIGN.comp.s_Wc', pt+'.comp.s_Wc')
    prob.model.connect('DESIGN.comp.s_eff', pt+'.comp.s_eff')
    prob.model.connect('DESIGN.comp.s_Nc', pt+'.comp.s_Nc')

    prob.model.connect('DESIGN.turb.s_PR', pt+'.turb.s_PR')
    prob.model.connect('DESIGN.turb.s_Wp', pt+'.turb.s_Wp')
    prob.model.connect('DESIGN.turb.s_eff', pt+'.turb.s_eff')
    prob.model.connect('DESIGN.turb.s_Np', pt+'.turb.s_Np')

    prob.model.connect('DESIGN.inlet.Fl_O:stat:area', pt+'.inlet.area')
    prob.model.connect('DESIGN.comp.Fl_O:stat:area', pt+'.comp.area')
    prob.model.connect('DESIGN.burner.Fl_O:stat:area', pt+'.burner.area')
    prob.model.connect('DESIGN.turb.Fl_O:stat:area', pt+'.turb.area')
    prob.model.connect('DESIGN.nozz.Throat:stat:area', pt+'.balance.rhs:W')

prob.setup(check=False)

# Set initial guesses for balances
prob['DESIGN.balance.FAR'] = 0.0175506829934
prob['DESIGN.balance.W'] = 168.453135137
prob['DESIGN.balance.turb_PR'] = 4.46138725662
prob['DESIGN.fc.balance.Pt'] = 14.6955113159
prob['DESIGN.fc.balance.Tt'] = 518.665288153

for pt in pts:
    prob[pt+'.balance.W'] = 166.073
    prob[pt+'.balance.FAR'] = 0.01680
    prob[pt+'.balance.Nmech'] = 8197.38
    prob[pt+'.fc.balance.Pt'] = 15.703
    prob[pt+'.fc.balance.Tt'] = 558.31
    prob[pt+'.turb.PR'] = 4.6690

st = time.time()

prob.set_solver_print(level=-1)
prob.set_solver_print(level=2, depth=1)
prob.run_model()

for pt in ['DESIGN']+pts:
    viewer(prob, pt)

print()
print("time", time.time() - st)





