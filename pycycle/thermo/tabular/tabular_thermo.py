import numpy as np
import openmdao.api as om
import pickle

from pycycle.constants import TAB_AIR_FUEL_COMPOSITION, AIR_JETA_TAB_SPEC


class Dummy(om.ExplicitComponent): 
    """ 
    dummy component to hold the composition 
    vector input so it can be promoted to that name
    """

    def initialize(self): 
        self.options.declare('composition')

    def setup(self): 
        size = len(self.options['composition'])

        self.add_input('composition', val=np.zeros(size))
        self.add_output('foo')

class SetTotalTP(om.Group):

    def initialize(self):
        self.options.declare('interp_method', default='slinear')
        self.options.declare('spec', recordable=False)
        self.options.declare('composition')

    def setup(self):
        interp_method = self.options['interp_method']
        spec = self.options['spec']
        composition = self.options['composition']

        # self.add_subsystem('dummy', Dummy(composition=composition), promotes_inputs=['composition'])

        thermo_data = pickle.load(open(spec, 'rb'))

        interp = om.MetaModelStructuredComp(method=interp_method)
        self.add_subsystem('tab', interp, promotes_inputs=['P', 'T'], 
                                          promotes_outputs=['h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'R'])

        for i, param in enumerate(composition): 
            interp.add_input(param, composition[param], training_data=thermo_data[param])
            self.promotes('tab', inputs=[(param, 'composition')], src_indices=[i,])

        interp.add_input('P', 101325.0, units='Pa', training_data=thermo_data['P'])
        interp.add_input('T', 273.0, units='degK', training_data=thermo_data['T'])

        interp.add_output('h', 0.0, units='J/kg', training_data=thermo_data['h'])
        interp.add_output('S', 0.0, units='J/kg/degK', training_data=thermo_data['S'])
        interp.add_output('gamma', 0.0, units=None, training_data=thermo_data['gamma'])
        interp.add_output('Cp', 0.0, units='J/kg/degK', training_data=thermo_data['Cp'])
        interp.add_output('Cv', 0.0, units='J/kg/degK', training_data=thermo_data['Cv'])
        interp.add_output('rho', 0.0, units='kg/m**3', training_data=thermo_data['rho'])
        interp.add_output('R', 0.0, units='J/kg/degK', training_data=thermo_data['R'])

        # self.set_input_defaults('composition', val=composition.values())

        # required part of the SetTotalTP API for flow setup
        self.composition = np.zeros(1) # not used for tabular
