import numpy as np
import openmdao.api as om
import pickle

from pycycle.thermo.cea import chem_eq as cea_thermo
from pycycle.thermo.tabular import tabular_thermo as tab_thermo
from pycycle.thermo.tabular import tab_thermo_gen as tab_thermo_gen
from pycycle.thermo.cea.species_data import janaf
from pycycle.constants import TAB_AIR_FUEL_COMPOSITION

p = om.Problem()
p.model = om.Group()

p.model.add_subsystem('tab', tab_thermo.SetTotalTP(thermo_spec='air_jetA.pkl', composition=TAB_AIR_FUEL_COMPOSITION), promotes_inputs=['*'])

p.model.add_subsystem('cea', tab_thermo_gen.TabThermoGenAirFuel(thermo_data=janaf, thermo_method='CEA'), promotes_inputs=['*'])

p.set_solver_print(level=-1)
p.setup()

p['FAR'] = 0.00
p['P'] = 101325 #7857143.1 
p['T'] = 500 #1977.8 


temp = np.random.rand(10,3)

for i, row in enumerate(temp):
    p['FAR'] = row[0]*(0.05)
    p['P'] = row[1]*1e6
    p['T'] = row[2]*(2500-150)+150
    print(p['FAR'], p['P'], p['T'])

    p.run_model()

    tab = p.get_val('tab.h')[0]
    cea = p.get_val('cea.flow:h', units='J/kg')[0]
    print('h:', abs((tab-cea)/cea))

    tab = p.get_val('tab.S')[0]
    cea = p.get_val('cea.flow:S', units='J/kg/degK')[0]
    print('S:', abs((tab-cea)/cea))

    tab = p.get_val('tab.gamma')[0]
    cea = p.get_val('cea.flow:gamma')[0]
    print('gamma:', abs((tab-cea)/cea))

    tab = p.get_val('tab.Cp')[0]
    cea = p.get_val('cea.flow:Cp', units='J/kg/degK')[0]
    print('Cp:', abs((tab-cea)/cea))

    tab = p.get_val('tab.Cv')[0]
    cea = p.get_val('cea.flow:Cv', units='J/kg/degK')[0]
    print('Cv:', abs((tab-cea)/cea))

    tab = p.get_val('tab.rho')[0]
    cea = p.get_val('cea.flow:rho', units='kg/m**3')[0]
    print('rho:', abs((tab-cea)/cea))
    
    tab = p.get_val('tab.R')[0]
    cea = p.get_val('cea.flow:R', units='J/kg/degK')[0]
    print('R:', abs((tab-cea)/cea))
    print()