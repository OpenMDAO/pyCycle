import numpy as np
import openmdao.api as om
import pickle

from pycycle.thermo.cea import chem_eq as cea_thermo
from pycycle.thermo.tabulated import tabulated_thermo as tab_thermo
from pycycle.thermo.tabulated import tab_thermo_gen as tab_thermo_gen
from pycycle.thermo.cea.species_data import janaf
from pycycle.constants import TAB_AIR_FUEL_COMPOSITION

thermo_data = pickle.load(open('air_jetA.pkl', 'rb'))

p = om.Problem()
p.model = om.Group()

p.model.add_subsystem('tab', tab_thermo.SetTotalTP(thermo_data=thermo_data, composition=TAB_AIR_FUEL_COMPOSITION), promotes_inputs=['*'])

p.model.add_subsystem('cea', tab_thermo_gen.TabThermoGenAir(thermo_data=janaf, thermo_method='CEA'), promotes_inputs=['*'])

p.set_solver_print(level=-1)
p.setup()

p['FAR'] = 0.00
p['P'] = 101325 #7857143.1 
p['T'] = 300 #1977.8 

p.run_model()

print('h:', p.get_val('tab.h')[0], p.get_val('cea.flow:h', units='J/kg')[0])
print('S:', p.get_val('tab.S')[0], p.get_val('cea.flow:S', units='J/kg/degK')[0])
print('gamma:', p.get_val('tab.gamma')[0], p.get_val('cea.flow:gamma')[0])
print('Cp:', p.get_val('tab.Cp')[0], p.get_val('cea.flow:Cp', units='J/kg/degK')[0])
print('Cv:', p.get_val('tab.Cv')[0], p.get_val('cea.flow:Cv', units='J/kg/degK')[0])
print('rho:', p.get_val('tab.rho')[0], p.get_val('cea.flow:rho', units='kg/m**3')[0])
print('R:', p.get_val('tab.R')[0], p.get_val('cea.flow:R', units='J/kg/degK')[0])
