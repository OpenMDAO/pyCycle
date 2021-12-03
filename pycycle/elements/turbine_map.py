import openmdao.api as om

from pycycle.maps.lpt2269 import LPT2269


class MapScalars(om.ExplicitComponent):
    """Compute map scalars"""

    def setup(self):
        self.add_input('eff', val=2.0,
                       desc='Design adiabatic efficiency')
        self.add_input('Np', val=2.0, units='rpm',
                       desc='Computed design referred shaft speed')
        self.add_input('PR', val=2.0, desc='Design pressure ratio')
        self.add_input('Wp', val=2.0, units='lbm/s',
                       desc='Computed design referred mass flow rate')
        self.add_input('effMap', val=2.0,
                       desc='Design adiabatic efficiency of map')
        self.add_input('NpMap', val=1.0, units='rpm',
                       desc='Design referred shaft speed of map')
        self.add_input('PRmap', val=1.0, desc='Design pressure ratio of map')
        self.add_input('WpMap', val=2.0, units='lbm/s',
                       desc='Design referred mass flow rate of map')

        self.add_output('s_eff', shape=1,
                        desc='Scalar for design adiabatic efficiency')
        self.add_output('s_Np', shape=1,
                        desc='Scalar for design corrected shaft speed')
        self.add_output('s_PR', shape=1,
                        desc='Scalar for design pressure ratio')
        self.add_output('s_Wp', shape=1,
                        desc='Scalar for design corrected mass flow rate')

        self.declare_partials('s_eff', ['eff', 'effMap'])
        self.declare_partials('s_Np', ['Np', 'NpMap'])
        self.declare_partials('s_PR', ['PR', 'PRmap'])
        self.declare_partials('s_Wp', ['Wp', 'WpMap'])

    def compute(self, inputs, outputs):
        outputs['s_Np'] = inputs['Np'] / inputs['NpMap']
        outputs['s_PR'] = (inputs['PR'] - 1) / (inputs['PRmap'] - 1)
        outputs['s_eff'] = inputs['eff'] / inputs['effMap']
        outputs['s_Wp'] = inputs['Wp'] / inputs['WpMap']

    def compute_partials(self, inputs, J):
        J['s_eff', 'eff'] = 1. / inputs['effMap']
        J['s_eff', 'effMap'] = - inputs['eff'] / inputs['effMap']**2
        J['s_Np', 'Np'] = 1. / inputs['NpMap']
        J['s_Np', 'NpMap'] = - inputs['Np'] / inputs['NpMap']**2
        J['s_PR', 'PR'] = 1. / (inputs['PRmap'] - 1)
        J['s_PR', 'PRmap'] = (1-inputs['PR']) / (inputs['PRmap'] - 1)**2
        J['s_Wp', 'Wp'] = 1. / inputs['WpMap']
        J['s_Wp', 'WpMap'] = -inputs['Wp'] / inputs['WpMap']**2

class ScaledMapValues(om.ExplicitComponent):
    """Scale map output"""

    def setup(self):
        self.add_input('effMap', val=1.0, desc='Efficiency from unscaled map')
        self.add_input('NpMap', val=1.0, units='rpm',
                       desc='Referred shaft speed from unscaled map')
        self.add_input('PRmap', val=1.0, desc='Pressure ratio from unscaled map')
        self.add_input('WpMap', val=1.0, units='lbm/s',
                       desc='Referred mass flow rate from unscaled map')
        self.add_input('s_eff', val=1.0,
                       desc='Scalar for adiabatic efficiency')
        self.add_input('s_Np', val=1.0,
                       desc='Scalar for referred shaft speed')
        self.add_input('s_PR', val=1.0, desc='Scalar for pressure ratio')
        self.add_input('s_Wp', val=1.0,
                       desc='Scalar for referred mass flow rate')

        self.add_output('eff', shape=1, desc='Adiabatic efficiency')
        self.add_output('Np', shape=1, units='rpm', desc='Referred shaft speed')
        self.add_output('PR', shape=1, desc='Pressure ratio')
        self.add_output('Wp', shape=1, units='lbm/s', desc='Referred mass flow rate')

        self.declare_partials('eff', ['effMap', 's_eff'])
        self.declare_partials('Np', ['NpMap', 's_Np'])
        self.declare_partials('PR', ['PRmap', 's_PR'])
        self.declare_partials('Wp', ['WpMap', 's_Wp'])

    def compute(self, inputs, outputs):
        outputs['eff'] = inputs['effMap'] * inputs['s_eff']
        outputs['Np'] = inputs['NpMap'] * inputs['s_Np']
        outputs['PR'] = ((inputs['PRmap'] - 1.) * inputs['s_PR']) + 1.
        outputs['Wp'] = inputs['WpMap'] * inputs['s_Wp']

    def compute_partials(self, inputs, J):
        J['eff', 'effMap'] = inputs['s_eff']
        J['eff', 's_eff'] = inputs['effMap']
        J['Np', 'NpMap'] = inputs['s_Np']
        J['Np', 's_Np'] = inputs['NpMap']
        J['PR', 'PRmap'] = inputs['s_PR']
        J['PR', 's_PR'] = inputs['PRmap'] - 1.
        J['Wp', 'WpMap'] = inputs['s_Wp']
        J['Wp', 's_Wp'] = inputs['WpMap']


class TurbineMap(om.Group):
    """runs design and off-design mode Turbine map calculations"""

    def initialize(self):
        self.options.declare('map_data', default=LPT2269)
        self.options.declare('design', default=True)
        self.options.declare('interp_method', default='slinear')
        self.options.declare('extrap', default=False)

    def setup(self):

        map_data = self.options['map_data']
        design = self.options['design']
        method = self.options['interp_method']
        extrap = self.options['extrap']

        params = map_data.param_data
        outputs = map_data.output_data

        # Define map which will be used
        readmap = om.MetaModelStructuredComp(method=method, extrapolate=extrap)
        for p in params:
            readmap.add_input(p['name'], val=p['default'], units=p['units'],
                        training_data=p['values'])
        for o in outputs:
            readmap.add_output(o['name'], val=o['default'], units=o['units'],
                        training_data=o['values'])

        if design:
            # In design mode, operating point specified by default values for RlineMap, NcMap and alphaMap
            self.set_input_defaults('NpMap', val=map_data.defaults['NpMap'], units='rpm')
            self.set_input_defaults('PRmap', val=map_data.defaults['PRmap'], units=None)

            # Evaluate map using design point values
            self.add_subsystem('readMap', readmap, promotes_inputs=['alphaMap', 'NpMap', 'PRmap'],
                                promotes_outputs=['effMap', 'WpMap'])

            # Compute map scalars based on input PR, eff, Np and Wp as well as unscaled map values
            self.add_subsystem('scalars', MapScalars(),
                                promotes_inputs=['PR', 'eff', 'Np', 'Wp', 'NpMap', 'effMap', 'PRmap', 'WpMap'],
                                promotes_outputs=['s_Np', 's_PR', 's_eff', 's_Wp'])

        else:
            # In off-design mode, PRmap, NpMap and alphaMap are input to map
            self.add_subsystem('readMap', readmap, promotes_inputs=['alphaMap', 'NpMap', 'PRmap'],
                                promotes_outputs=['effMap', 'WpMap'])

            # Compute scaled map outputs base on input scalars and unscaled map values
            self.add_subsystem('scaledOutput', ScaledMapValues(),
                                promotes_inputs=['s_PR', 's_eff', 's_Wp', 's_Np', 'NpMap', 'effMap', 'PRmap', 'WpMap'],
                                promotes_outputs=['PR', 'eff'])

            # Use balance component to vary NpMap and PRmap to match incoming corrected flow and speed
            map_bal = om.BalanceComp()
            map_bal.add_balance('NpMap', val=map_data.defaults['NpMap'], units='rpm', eq_units='rpm', lower=.1, upper=200.)
            map_bal.add_balance('PRmap', val=map_data.defaults['PRmap'], units=None,
                                eq_units='lbm/s', lower=1.01)
            self.add_subsystem(name='map_bal', subsys=map_bal,
                                promotes_inputs=[('lhs:NpMap','Np'),('lhs:PRmap','Wp')],
                                promotes_outputs=['NpMap', 'PRmap'])
            self.connect('scaledOutput.Np','map_bal.rhs:NpMap')
            self.connect('scaledOutput.Wp','map_bal.rhs:PRmap')


if __name__ == "__main__":
    from pycycle.maps.lpt2269 import LPT2269

    p = om.Problem()
    des_vars = p.model.add_subsystem(
        'des_vars', om.IndepVarComp(), promotes=['*'])
    des_vars.add_output('Wp', 322.60579101811692)
    des_vars.add_output('Np', 172.11870165984794)
    des_vars.add_output('alphaMap', 1.5)
    des_vars.add_output('s_NpDes', 1.721074624)
    des_vars.add_output('s_PRdes', 0.147473296)
    des_vars.add_output('s_WpDes', 2.152309293)
    des_vars.add_output('s_effDes', 0.9950409659)

    p.model.add_subsystem('map', TurbineMap(
        map_data=LPT2269, design=True), promotes=['*'])

    # Target PR: 1.7373664799999999

    p.setup(check=True)
    print(p['Wp'], p['Wp'])
    p.run_model()
    p.check_partials(compact_print=False)
    # these should match
    print(p['Wp'], p['Wp'])

    # this should be 1.7373664799999999 ish
    print(p['PR'])

    print(p['NpMap'])
    print(p['PRmap'])
