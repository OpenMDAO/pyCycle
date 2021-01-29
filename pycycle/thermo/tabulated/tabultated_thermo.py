import numpy as np
import openmdao.api as om
import pickle

class SetTotalTP(om.Group):

    def initialize(self):
        self.options.declare('interp_method', default='scipy_cubic')
        self.options.declare('thermo_data', recordable=False)
        self.options.declare('composition')

    def setup(self):
        pass
        interp_method = self.options['interp_method']
        thermo_data = self.options['thermo_data']

        interp = om.MetaModelStructuredComp(method=interp_method)
        # in self.options['composition']: 
        #     interp.add_input(param, 0.0, training_data=thermo_data.XXXXXX)

        interp.add_input('FAR', 0.0, units='Pa', training_data=thermo_data['FAR'])    
        interp.add_input('P', 0.0, units='Pa', training_data=thermo_data['P'])
        interp.add_input('T', 0.5, units='degK', training_data=thermo_data['T'])

        interp.add_output('h', 0.0, units='J/kg', training_data=thermo_data['h'])
        interp.add_output('S', 0.0, units='J/kg/degK', training_data=thermo_data['S'])
        interp.add_output('gamma', 0.0, units=None, training_data=thermo_data['gamma'])
        interp.add_output('Cp', 0.0, units='J/kg/degK', training_data=thermo_data['Cp'])
        interp.add_output('Cv', 0.0, units='J/kg/degK', training_data=thermo_data['Cv'])
        interp.add_output('rho', 0.0, units='kg/m**3', training_data=thermo_data['rho'])
        interp.add_output('R', 0.0, units='J/kg/degK', training_data=thermo_data['R'])

        self.add_subsystem('tab', interp, promotes=['*'])

if __name__ == '__main__':

    thermo_data = pickle.load(open('air_jetA.pkl', 'rb'))

    p = om.Problem()
    p.model = SetTotalTP(thermo_data=thermo_data)

    p.setup()

    p['FAR'] = 0.03
    p['P'] = 101325*3
    p['T'] = 1500

    p.run_model()

    print('h:', p.get_val('h')[0])
    print('S:', p.get_val('S')[0])
    print('gamma:', p.get_val('gamma')[0])
    print('Cp:', p.get_val('Cp')[0])
    print('Cv:', p.get_val('Cv')[0])
    print('rho:', p.get_val('rho')[0])
    print('R:', p.get_val('R')[0])