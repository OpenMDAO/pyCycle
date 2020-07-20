import openmdao.api as om

from pycycle.constants import AIR_MIX
from pycycle.cea.set_total import SetTotal
from pycycle.cea.unit_comps import EngUnitStaticProps, EngUnitProps
from pycycle.cea import species_data

class SetStatic(om.Group):

    def initialize(self):
        self.options.declare('mode', values=['Ps', 'area', 'MN'])
        self.options.declare('thermo_data', desc='thermodynamic data set', recordable=False)
        self.options.declare('fl_name',
                              default="flow",
                              desc='flowstation name of the output flow variables')
        self.options.declare('init_reacts',
                              default=AIR_MIX,
                              desc='initial amounts of each species in the flow')

    def setup(self):

        mode = self.options['mode']

        thermo_data = self.options['thermo_data']
        init_reacts = self.options['init_reacts']
        fl_name = self.options['fl_name']

        thermo = species_data.Thermo(thermo_data, init_reacts)

        statics = SetTotal(mode='S',
                           fl_name=fl_name,
                           thermo_data=thermo_data,
                           init_reacts=init_reacts,
                           for_statics=mode)

        # have to promote things differently depending on which mode we are
        if mode == 'Ps':
            self.add_subsystem('statics', statics,
                               promotes_inputs=[('P', 'Ps'), 'S', 'ht', 'W', 'init_prod_amounts'],
                               promotes_outputs=['MN', 'V', 'Vsonic', 'area',
                                                 'T', 'h', 'gamma', 'Cp', 'Cv', 'rho', 'n', 'n_moles'])
        elif mode == 'MN':
            self.add_subsystem('statics', statics,
                               promotes_inputs=['MN', 'S', 'ht', 'W', 'guess:*', 'init_prod_amounts'],
                               promotes_outputs=['V', 'Vsonic', 'area',
                                                 'Ps', 'T', 'h', 'gamma', 'Cp', 'Cv', 'rho', 'n', 'n_moles'])

        else:
            self.add_subsystem('statics', statics,
                               promotes_inputs=['area', 'S', 'ht', 'W', 'guess:*', 'init_prod_amounts'],
                               promotes_outputs=['V', 'Vsonic', 'MN',
                                                 'Ps', 'T', 'h', 'gamma', 'Cp', 'Cv', 'rho', 'n', 'n_moles'])

        p_inputs = ('T', 'P', 'h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'n', 'n_moles')
        p_outputs = tuple(['{0}:{1}'.format(fl_name, in_name) for in_name in p_inputs])
        # need to redefine this so that P gets promoted as P. Needed the first definition for the list comprehension
        p_inputs = ('T', ('P', 'Ps'), 'h', 'S', 'gamma', 'Cp', 'Cv', 'rho', 'n', 'n_moles')

        self.add_subsystem('flow', EngUnitProps(thermo=thermo, fl_name=fl_name),
                           promotes_inputs=p_inputs,
                           promotes_outputs=p_outputs)

        p_inputs = ('area', 'W', 'V', 'Vsonic', 'MN')
        p_outputs = tuple(['{0}:{1}'.format(fl_name, in_name) for in_name in p_inputs])
        eng_units_statics = EngUnitStaticProps(thermo, fl_name)
        self.add_subsystem('flow_static', eng_units_statics,
                           promotes_inputs=p_inputs,
                           promotes_outputs=p_outputs)

        self.set_input_defaults('area', units='m**2', val=1.)
        # self.set_order(['statics', 'flow', 'flow_static'])

if __name__ == "__main__":


    p = om.Problem()
    indeps = p.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])
    indeps.add_output('T', 1500, units="degK")
    indeps.add_output('P', 1.034210, units="bar")
