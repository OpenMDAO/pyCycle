import numpy as np
import openmdao.api as om
import pickle

from pycycle.constants import TAB_AIR_FUEL_COMPOSITION, AIR_JETA_TAB_SPEC


class SetTotalTP(om.Group):

    def initialize(self):
        self.options.declare('interp_method', default='slinear')
        self.options.declare('spec', types=str)
        self.options.declare('composition')

    def setup(self):
        interp_method = self.options['interp_method']
        spec = self.options['spec']
        composition = self.options['composition']

        sorted_comp = sorted(composition.keys())

        with open(spec, 'rb') as spec_data:
            thermo_data = pickle.load(spec_data)

        interp = om.MetaModelStructuredComp(method=interp_method, extrapolate=True)
        self.add_subsystem('tab', interp, promotes_inputs=['P', 'T'], 
                                          promotes_outputs=['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R'])

        for i, param in enumerate(sorted_comp): 
            interp.add_input(param, composition[param], training_data=thermo_data[param])
            self.promotes('tab', inputs=[(param, 'composition')], src_indices=[i,])
        self.set_input_defaults('composition', src_shape=len(composition))

        interp.add_input('P', 101325.0, units='Pa', training_data=thermo_data['P'])
        interp.add_input('T', 273.0, units='degK', training_data=thermo_data['T'])

        interp.add_output('h', 0.0, units='J/kg', training_data=thermo_data['h'])
        interp.add_output('S', 0.0, units='J/kg/degK', training_data=thermo_data['S'])
        interp.add_output('gamma', 1.4, units=None, training_data=thermo_data['gamma'])
        interp.add_output('Cp', 0.0, units='J/kg/degK', training_data=thermo_data['Cp'])
        interp.add_output('Cv', 0.0, units='J/kg/degK', training_data=thermo_data['Cv'])
        interp.add_output('rho', 1.0, units='kg/m**3', training_data=thermo_data['rho'])
        interp.add_output('R', 287.0, units='J/kg/degK', training_data=thermo_data['R'])

        # required part of the SetTotalTP API for flow setup
        # pass the sorted list of keys from the composition dictionary downstream, so all comps use the same ordering
        self.composition = sorted_comp
