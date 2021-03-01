"""
Sample script showing how to generate tabular thermodynamic
data for use with tabular thermodynamics

This scrip generates a pickle file called 'air_jetA.pkl' which is 
equivalent to the default tabular thermo data in pyCycle. 

You can generate a custom pickle file, then provide the path to that file 
as the thermo_spec for tabular therm
"""

import numpy as np
import openmdao.api as om
import pickle

from pycycle.thermo.thermo import Thermo, ThermoAdd
from pycycle.constants import CEA_AIR_COMPOSITION, CEA_AIR_FUEL_COMPOSITION, ALLOWED_THERMOS
from pycycle.thermo.cea.species_data import janaf, wet_air


class TabThermoGenAir(om.Group):

    def initialize(self):
        self.options.declare('thermo_data', default=False,
                              desc='thermodynamic data specific to this element', recordable=False)
        self.options.declare('thermo_method', default='CEA', values=ALLOWED_THERMOS,
                              desc='Method for computing thermodynamic properties')

    def setup(self):

        thermo_data = self.options['thermo_data']
        thermo_method = self.options['thermo_method']

        flow = Thermo(mode='total_TP', fl_name='flow', 
                          method=thermo_method, 
                          thermo_kwargs={'composition':CEA_AIR_COMPOSITION, 
                                         'spec':thermo_data})
        self.add_subsystem('flow', flow, promotes_inputs=['T','P'],
                                        promotes_outputs=['flow:*'])

        self.set_input_defaults('T', val=273.15, units='degK')
        self.set_input_defaults('P', val=101325, units='Pa')


class TabThermoGenAirFuel(om.Group):

    def initialize(self):
        self.options.declare('fuel_type', default="Jet-A(g)",
                             desc='Type of fuel.')
        self.options.declare('thermo_data', default=False,
                              desc='thermodynamic data specific to this element', recordable=False)
        self.options.declare('thermo_method', default='CEA', values=ALLOWED_THERMOS,
                              desc='Method for computing thermodynamic properties')


    def setup(self):

        fuel_type = self.options['fuel_type']
        thermo_data = self.options['thermo_data']
        thermo_method = self.options['thermo_method']

        # Compute mixed air-fuel composition
        self.thermo_add_comp = ThermoAdd(method=thermo_method, mix_mode='reactant',
                                         thermo_kwargs={'spec':thermo_data,
                                                        'inflow_composition':CEA_AIR_COMPOSITION, 
                                                        'mix_composition':fuel_type})
        self.add_subsystem('mix_fuel', self.thermo_add_comp,
                           promotes=[('Fl_I:stat:W','W'), ('mix:ratio', 'FAR'), ('Fl_I:tot:composition','composition'),
                                     ('Fl_I:tot:h','h'), ('mix:W','Wfuel'), 'Wout'])

        # Compute properties of air-fuel mixture
        vit_flow = Thermo(mode='total_TP', fl_name='flow', 
                          method=thermo_method, 
                          thermo_kwargs={'composition':CEA_AIR_FUEL_COMPOSITION, 
                                         'spec':thermo_data})
        self.add_subsystem('vitiated_flow', vit_flow, promotes_inputs=['T','P'],
                                        promotes_outputs=['flow:*'])
        self.connect("mix_fuel.composition_out", "vitiated_flow.composition")

        self.set_input_defaults('W', val=1.0, units='kg/s')
        self.set_input_defaults('FAR', val=0.02)
        self.set_input_defaults('T', val=273.15, units='degK')
        self.set_input_defaults('P', val=101325, units='Pa')


if __name__ == "__main__":

    # FAR - lower: 0.0,  upper 0.05
    # P - lower: 0.886280 Pa,  upper: 10132500 Pa
    # T - lower: 196.650 degK,  upper: 2500 degK

    # FAR_range = [0, 0.03]
    # P_range = [1e6]
    # T_range = [1500]

    FAR_range = np.linspace(0.0, 0.05, num=20)
    # WAR_range = np.linspace(0.0, 1.0, num=5)
    # P_range = np.linspace(1, 1e7, num=30)
    P_range = np.logspace(0, 7, num=110)
    T_range = np.linspace(100, 3500, num=100)
    numFAR = len(FAR_range)
    numP = len(P_range)
    numT = len(T_range)

    h = np.empty([numFAR,numP,numT])
    S = np.empty([numFAR,numP,numT])
    gamma = np.empty([numFAR,numP,numT])
    Cp = np.empty([numFAR,numP,numT])
    Cv = np.empty([numFAR,numP,numT])
    rho = np.empty([numFAR,numP,numT])
    R = np.empty([numFAR,numP,numT])


    #############################################
    # Pure Air  - Run this for FAR=0 cases
    #############################################

    p = om.Problem()
    p.model = TabThermoGenAir(thermo_data=janaf, thermo_method='CEA')

    p.setup(check=False)
    p.set_solver_print(level=-1)

    for j, P in enumerate(P_range):
        p['P'] = P
        for k, T in enumerate(T_range):
            p['T'] = T
            p.run_model()

            print('FAR: %.3f, P: %.1f, T: %.1f' %(0.0,p['P'][0],p['T'][0]))
            h[0,j,k] = p.get_val('flow:h', units='J/kg')[0]
            S[0,j,k] = p.get_val('flow:S', units='J/kg/degK')[0]
            gamma[0,j,k] = p.get_val('flow:gamma')[0]
            Cp[0,j,k] = p.get_val('flow:Cp', units='J/kg/degK')[0]
            Cv[0,j,k] = p.get_val('flow:Cv', units='J/kg/degK')[0]
            rho[0,j,k] = p.get_val('flow:rho', units='kg/m**3')[0]
            R[0,j,k] = p.get_val('flow:R', units='J/kg/degK')[0]


    #############################################
    # Vitiated Air  - Run this for FAR>0 cases
    #############################################
    p2 = om.Problem()
    p2.model = TabThermoGenAirFuel(fuel_type='Jet-A(g)', thermo_data=wet_air, thermo_method='CEA')

    p2.setup(check=False)
    p2.set_solver_print(level=-1)

    for i, FAR in enumerate(FAR_range):
        if i==0:
            pass
        else:
            p2['FAR'] = FAR
            for j, P in enumerate(P_range):
                p2['P'] = P
                for k, T in enumerate(T_range):
                    p2['T'] = T
                    p2.run_model()

                    print('FAR: %.3f, P: %.1f, T: %.1f' %(p2['FAR'][0],p2['P'][0],p2['T'][0]))
                    h[i,j,k] = p2.get_val('flow:h', units='J/kg')[0]
                    S[i,j,k] = p2.get_val('flow:S', units='J/kg/degK')[0]
                    gamma[i,j,k] = p2.get_val('flow:gamma')[0]
                    Cp[i,j,k] = p2.get_val('flow:Cp', units='J/kg/degK')[0]
                    Cv[i,j,k] = p2.get_val('flow:Cv', units='J/kg/degK')[0]
                    rho[i,j,k] = p2.get_val('flow:rho', units='kg/m**3')[0]
                    R[i,j,k] = p2.get_val('flow:R', units='J/kg/degK')[0]

    thermo_data_dict = {'T':T_range, 'P':P_range, 'FAR':FAR_range, 'h':h, 'S':S, 'gamma':gamma, 'Cp':Cp,
                        'Cv':Cv, 'rho':rho, 'R':R}


    pickle.dump( thermo_data_dict, open('air_jetA.pkl', 'wb'))
