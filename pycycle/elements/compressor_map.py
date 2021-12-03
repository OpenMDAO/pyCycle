import openmdao.api as om

from pycycle.maps.ncp01 import NCP01

import numpy as np


class StallCalcs(om.ExplicitComponent):
    """Component to compute the stall margins at constant speed (SMN) and constant flow (SMW)"""

    def setup(self):

        self.add_input('PR_SMN', val=1.0, units=None, desc='SMN pressure ratio')
        self.add_input('PR_SMW', val=1.0, units=None, desc='SMW pressure ratio')
        self.add_input('PR_actual', val=1.0, units=None, desc='Actual pressure ratio')

        self.add_input('Wc_SMN', val=1.0, units='lbm/s', desc='SMN corrected flow')
        self.add_input('Wc_actual', val=1.0, units='lbm/s', desc='Actual corrected flow')

        self.add_output('SMN', val=0.0, units=None, desc='Stall margin at constant speed')
        self.add_output('SMW', val=0.0, units=None, desc='Stall margin at constant flow')

        self.declare_partials('SMN', ['PR_SMN','PR_actual','Wc_SMN','Wc_actual'])
        self.declare_partials('SMW', ['PR_SMW','PR_actual'])

    def compute(self, inputs, outputs):

        outputs['SMN'] = ((inputs['Wc_actual']/inputs['Wc_SMN'])/(inputs['PR_actual']/inputs['PR_SMN'])-1)*100.
        outputs['SMW'] = (inputs['PR_SMW']-inputs['PR_actual'])/inputs['PR_actual'] * 100

    def compute_partials(self, inputs, J):

        PR_actual = inputs['PR_actual']
        PR_SMN = inputs['PR_SMN']
        PR_SMW = inputs['PR_SMW']

        Wc_actual = inputs['Wc_actual']
        Wc_SMN = inputs['Wc_SMN']

        wc_ratio = 100*Wc_actual/Wc_SMN
        J['SMN', 'PR_SMN'] = wc_ratio / PR_actual
        J['SMN', 'PR_actual'] = -wc_ratio * PR_SMN/PR_actual**2

        PR_ratio = 100*PR_SMN/PR_actual
        J['SMN', 'Wc_SMN'] = -Wc_actual/Wc_SMN**2 * PR_ratio
        J['SMN', 'Wc_actual'] = PR_ratio /Wc_SMN

        J['SMW', 'PR_SMW'] = 100 / PR_actual
        J['SMW', 'PR_actual'] = - 100 * PR_SMW/PR_actual**2



class MapScalars(om.ExplicitComponent):
    """Compute map scalars in design mode"""

    def setup(self):

        self.add_input('Nc', val=2.0, units='rpm',
                       desc='Computed design corrected shaft speed')
        self.add_input('NcMap', val=2.0, units='rpm',
                       desc='Design corrected shaft speed of map')
        self.add_input('PR', val=2.0,
                       desc='User input design pressure ratio')
        self.add_input('PRmap', val=2.0,
                       desc='Design pressure ratio of map')
        self.add_input('eff', val=1.0,
                       desc='User input design adiabatic efficiency')
        self.add_input('effMap', val=1.0,
                       desc='Design adiabatic efficiency of map')
        self.add_input('Wc', val=2.0, units='lbm/s',
                       desc='Computed design corrected mass flow rate')
        self.add_input('WcMap', val=2.0, units='lbm/s',
                       desc='Design corrected mass flow rate of map')

        self.add_output('s_Nc', shape=1,
                        desc='Scalar for design corrected shaft speed')
        self.add_output('s_PR', shape=1,
                        desc='Scalar for design pressure ratio')
        self.add_output('s_eff', shape=1,
                        desc='Scalar for design adiabatic efficiency')
        self.add_output('s_Wc', shape=1,
                        desc='Scalar for design corrected mass flow rate')

        self.declare_partials('s_Nc', ['Nc', 'NcMap'])
        self.declare_partials('s_PR', ['PR', 'PRmap'])
        self.declare_partials('s_eff', ['eff', 'effMap'])
        self.declare_partials('s_Wc', ['Wc', 'WcMap'])

    def compute(self, inputs, outputs):
        outputs['s_Nc'] = inputs['Nc'] / inputs['NcMap']
        outputs['s_PR'] = (inputs['PR'] - 1) / (inputs['PRmap'] - 1)
        outputs['s_eff'] = inputs['eff'] / inputs['effMap']
        outputs['s_Wc'] = inputs['Wc'] / inputs['WcMap']

    def compute_partials(self, inputs, J):
        J['s_Nc', 'Nc'] = 1. / inputs['NcMap']
        J['s_Nc', 'NcMap'] = -inputs['Nc'] / inputs['NcMap']**2
        J['s_PR', 'PR'] = 1. / (inputs['PRmap'] - 1)
        J['s_PR', 'PRmap'] = -(inputs['PR'] - 1.) / (inputs['PRmap'] - 1.)**2
        J['s_eff', 'eff'] = 1. / inputs['effMap']
        J['s_eff', 'effMap'] = -inputs['eff'] * inputs['effMap']**(-2)
        J['s_Wc', 'Wc'] = 1. / inputs['WcMap']
        J['s_Wc', 'WcMap'] = -inputs['Wc'] * inputs['WcMap']**(-2)


class ScaledMapValues(om.ExplicitComponent):
    """Computes scaled map values for off-design mode"""


    def setup(self):

        self.add_input('effMap', val=2.0, desc='Efficiency from unscaled map')
        self.add_input('PRmap', val=2.0,
                       desc='Pressure ratio from unscaled map')
        self.add_input('WcMap', val=2.0, units='lbm/s',
                       desc='Corrected mass flow rate from unscaled map')
        self.add_input('NcMap', val=2.0, units='rpm',
                       desc='Corrected shaft speed from unscaled map')
        self.add_input('s_PR', val=2.0,
                       desc='Scalar for design corrected pressure ratio')
        self.add_input('s_eff', val=2.0,
                       desc='Scalar for design corrected adiabatic efficiency')
        self.add_input('s_Wc', val=2.0,
                       desc='Scalar for design corrected mass flow rate')
        self.add_input('s_Nc', val=2.0,
                       desc='Scalar for design corrected speed')

        self.add_output('PR', shape=1, desc='Pressure ratio', lower=1.00001)
        self.add_output('eff', shape=1, desc='Adiabatic efficiency')
        self.add_output('Wc', shape=1,
                        desc='Corrected mass flow rate', units='lbm/s')
        self.add_output('Nc', shape=1,
                        desc='Corrected shaft speed', units='rpm')

        self.declare_partials('PR', ['PRmap', 's_PR'])
        self.declare_partials('eff', ['effMap', 's_eff'])
        self.declare_partials('Wc', ['WcMap', 's_Wc'])
        self.declare_partials('Nc', ['NcMap', 's_Nc'])

    def compute(self, inputs, outputs):
        outputs['PR'] = (inputs['PRmap'] - 1.) * inputs['s_PR'] + 1.
        outputs['eff'] = inputs['effMap'] * inputs['s_eff']
        outputs['Wc'] = inputs['WcMap'] * inputs['s_Wc']
        outputs['Nc'] = inputs['NcMap'] * inputs['s_Nc']

    def compute_partials(self, inputs, J):
        J['PR', 'PRmap'] = inputs['s_PR']
        J['PR', 's_PR'] = inputs['PRmap'] - 1.
        J['eff', 'effMap'] = inputs['s_eff']
        J['eff', 's_eff'] = inputs['effMap']
        J['Wc', 'WcMap'] = inputs['s_Wc']
        J['Wc', 's_Wc'] = inputs['WcMap']
        J['Nc', 'NcMap'] = inputs['s_Nc']
        J['Nc', 's_Nc'] = inputs['NcMap']


class CompressorMap(om.Group):
    """Runs design and off-design mode compressor map calculations"""

    def initialize(self):
        self.options.declare('map_data', default=NCP01)
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
            readmap.add_input(p['name'], val=p['default'], units=p['units'], training_data=p['values'])
        for o in outputs:
            readmap.add_output(o['name'], val=o['default'], units=o['units'], training_data=o['values'])

        # Create instance of map for evaluating actual operating point
        if design:
            # In design mode, operating point specified by default values for RlineMap, NcMap and alphaMap
            self.set_input_defaults('RlineMap', val=map_data.defaults['RlineMap'], units=None)
            self.set_input_defaults('NcMap', val=map_data.defaults['NcMap'], units='rpm')

            # Evaluate map using design point values
            self.add_subsystem('map', readmap, promotes_inputs=['RlineMap', 'NcMap', 'alphaMap'],
                                promotes_outputs=['effMap', 'PRmap', 'WcMap'])

            # Compute map scalars based on input PR, eff, Nc and Wc as well as unscaled map values
            self.add_subsystem('scalars', MapScalars(),
                                promotes_inputs=['PR', 'eff', 'Nc', 'Wc', 'NcMap', 'effMap', 'PRmap', 'WcMap'],
                                promotes_outputs=['s_Nc', 's_PR', 's_eff', 's_Wc'])

        else:
            # In off-design mode, RlineMap, NcMap and alphaMap are input to map
            self.add_subsystem('map', readmap, promotes_inputs=['RlineMap', 'NcMap', 'alphaMap'],
                                promotes_outputs=['effMap', 'PRmap', 'WcMap'])

            # Compute scaled map outputs base on input scalars and unscaled map values
            self.add_subsystem('scaledOutput', ScaledMapValues(),
                                promotes_inputs=['s_PR', 's_eff', 's_Wc', 's_Nc', 'NcMap', 'effMap', 'PRmap', 'WcMap'],
                                promotes_outputs=['PR', 'eff'])

            # Use balance component to vary NcMap and RlineMap to match incoming corrected flow and speed
            map_bal = om.BalanceComp()
            map_bal.add_balance('NcMap', val=map_data.defaults['NcMap'], units='rpm', eq_units='rpm')
            map_bal.add_balance('RlineMap', val=map_data.defaults['RlineMap'], units=None, 
                                eq_units='lbm/s', lower=map_data.RlineStall)
            self.add_subsystem(name='map_bal', subsys=map_bal, 
                                promotes_inputs=[('lhs:NcMap','Nc'),('lhs:RlineMap','Wc')],
                                promotes_outputs=['NcMap', 'RlineMap'])
            self.connect('scaledOutput.Nc','map_bal.rhs:NcMap')
            self.connect('scaledOutput.Wc','map_bal.rhs:RlineMap')

        # Define the Rline corresponding to stall
        RlineStall = om.IndepVarComp()
        RlineStall.add_output('RlineStall', val=map_data.RlineStall, units=None)
        self.add_subsystem('stall_R', subsys=RlineStall)

        # Evaluate map for the constant speed stall margin (SMN)
        SMN_map = om.MetaModelStructuredComp(method=method, extrapolate=extrap)
        for p in params:
            SMN_map.add_input(p['name'], val=p['default'], units=p['units'], training_data=p['values'])
        for o in outputs:
            SMN_map.add_output(o['name'], val=o['default'], units=o['units'], training_data=o['values'])

        self.add_subsystem('SMN_map', SMN_map, promotes_inputs=['NcMap', 'alphaMap'])
        self.connect('stall_R.RlineStall', 'SMN_map.RlineMap')

        # Evaluate map for the constant speed stall margin (SMN)
        SMW_map = om.MetaModelStructuredComp(method=method, extrapolate=extrap)
        for p in params:
            SMW_map.add_input(p['name'], val=p['default'], units=p['units'], training_data=p['values'])
        for o in outputs:
            SMW_map.add_output(o['name'], val=o['default'], units=o['units'], training_data=o['values'])
        self.add_subsystem('SMW_map', SMW_map, promotes_inputs=['alphaMap'])
        self.connect('stall_R.RlineStall', 'SMW_map.RlineMap')

        # Use balance to vary NcMap on SMW map to hold corrected flow constant
        SMW_bal = om.BalanceComp()
        SMW_bal.add_balance('NcMap', val=map_data.defaults['NcMap'], units='rpm', eq_units='lbm/s')
        self.add_subsystem(name='SMW_bal', subsys=SMW_bal)
        self.connect('SMW_bal.NcMap', 'SMW_map.NcMap')
        self.connect('WcMap','SMW_bal.lhs:NcMap')
        self.connect('SMW_map.WcMap','SMW_bal.rhs:NcMap')

        # Compute the stall margins
        self.add_subsystem('stall_margins', StallCalcs(), 
                                promotes_inputs=[('PR_actual','PRmap'),('Wc_actual','WcMap')],
                                promotes_outputs=['SMN','SMW'])
        self.connect('SMN_map.PRmap', 'stall_margins.PR_SMN')
        self.connect('SMW_map.PRmap', 'stall_margins.PR_SMW')
        self.connect('SMN_map.WcMap', 'stall_margins.Wc_SMN')




if __name__ == "__main__":
    from pycycle.maps.ncp01 import NCP01

    p1 = om.Problem()
    ivc = p1.model.add_subsystem(
        'ivc', om.IndepVarComp(), promotes=['*'])
    # Design variables
    ivc.add_output('alphaMap', 0.0)
    ivc.add_output('PR', 2.0)
    ivc.add_output('Nc', 1000.0, units='rpm')
    ivc.add_output('eff', .9)
    ivc.add_output('Wc', 3000., units='lbm/s')

    # Off-design variables
    # ivc.add_output('alphaMap', 0.0)
    # ivc.add_output('Nc', 1000.0, units='rpm')
    # ivc.add_output('Wc', 3000., units='lbm/s')
    # ivc.add_output('s_Nc', 1.0)
    # ivc.add_output('s_Wc', 1.0)
    # ivc.add_output('s_PR', 1.0)
    # ivc.add_output('s_eff', 1.0)

    p1.model.add_subsystem('map', CompressorMap(
        map_data=NCP01, design=True), promotes=['*'])
    p1.setup()

    # exit()

    p1.run_model()
    p1.check_partials()

    # print('s_PRdes: ', p1['s_PRdes'])
    # print('s_effDes: ', p1['s_effDes'])
    # print('s_NcDes: ', p1['s_NcDes'])
    # print('s_WcDes: ', p1['s_WcDes'])
    # print(p1['shaftNc.NcMap'])
    # print(p1['readMap.PRmap'])
    # print(p1['readMap.effMap'])
    # print(p1['readMap.WcMap'])

    # print('Eff: ', p1['eff'])
    # print('PR: ', p1['PR'])
    # print('Wc: ', p1['Wc'])

    # p2 = Problem()
    # p2.model = CompressorMap(map_data=NCP01, design=False)
    # p2.setup()

    # p2['s_PRdes'] = 2.0
    # p2['s_NcDes'] = 1000.0
    # p2['s_effDes'] = 0.983606557377
    # p2['s_WcDes'] = 0.937500146484
    # p2['Nc'] = 900.0
    # p2['RlineMap'] = 2.0
    # p2['alphaMap'] = 0.0

    # p2.run_model()

    # print('s_PRdes: ', p2['s_PRdes'])
    # print('s_effDes: ', p2['s_effDes'])
    # print('s_NcDes: ', p2['s_NcDes'])
    # print('s_WcDes: ', p2['s_WcDes'])
    # print(p2['shaftNc.NcMap'])
    # print(p2['readMap.PRmap'])
    # print(p2['readMap.effMap'])
    # print(p2['readMap.WcMap'])

    # print('Eff: ', p2['eff'])
    # print('PR: ', p2['PR'])
    # print('Wc: ', p2['Wc'])
