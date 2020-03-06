import sys
import numpy as np

import openmdao.api as om

import pycycle.api as pyc


class MixedFlowTurbofan(om.Group):

    def initialize(self):
        self.options.declare('design', default=True,
            desc='Switch between on-design and off-design calculation.')

    def setup(self):
        thermo_spec = pyc.species_data.janaf
        design = self.options['design']

        self.add_subsystem('fc', pyc.FlightConditions(thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        # Inlet Components
        self.add_subsystem('inlet', pyc.Inlet(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('inlet_duct', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        # Fan Components - Split here for CFD integration Add a CFDStart Compomponent
        self.add_subsystem('fan', pyc.Compressor(map_data=pyc.AXI5, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX,
                                             map_extrap=True),promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('splitter', pyc.Splitter(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        # Core Stream components
        self.add_subsystem('splitter_core_duct', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('lpc', pyc.Compressor(map_data=pyc.LPCMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX,map_extrap=True),
                                             promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('lpc_duct', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('hpc', pyc.Compressor(map_data=pyc.HPCMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX,
                                        bleed_names=['cool1'],map_extrap=True),promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('bld3', pyc.BleedOut(design=design, bleed_names=['cool3']))
        self.add_subsystem('burner', pyc.Combustor(design=design,thermo_data=thermo_spec,
                                                inflow_elements=pyc.AIR_MIX,
                                                air_fuel_elements=pyc.AIR_FUEL_MIX,
                                                fuel_type='Jet-A(g)'))
        self.add_subsystem('hpt', pyc.Turbine(map_data=pyc.HPTMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX,
                                          bleed_names=['cool3'],map_extrap=True),promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('hpt_duct', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.add_subsystem('lpt', pyc.Turbine(map_data=pyc.LPTMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX,
                                        bleed_names=['cool1'],map_extrap=True), promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('lpt_duct', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        # Bypass Components
        self.add_subsystem('bypass_duct', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        # Mixer component
        self.add_subsystem('mixer', pyc.Mixer(design=design, designed_stream=1, Fl_I1_elements=pyc.AIR_FUEL_MIX, Fl_I2_elements=pyc.AIR_MIX))
        self.add_subsystem('mixer_duct', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        # Afterburner Components
        self.add_subsystem('afterburner', pyc.Combustor(design=design,thermo_data=thermo_spec,
                                                inflow_elements=pyc.AIR_FUEL_MIX,
                                                air_fuel_elements=pyc.AIR_FUEL_MIX,
                                                fuel_type='Jet-A(g)'))
        # End CFD HERE
        # Nozzle
        self.add_subsystem('mixed_nozz', pyc.Nozzle(nozzType='CD', lossCoef='Cfg', thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))

        # Mechanical components
        self.add_subsystem('lp_shaft', pyc.Shaft(num_ports=3),promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('hp_shaft', pyc.Shaft(num_ports=2),promotes_inputs=[('Nmech','HP_Nmech')])

        # Aggregating component
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=2))

        # Connnect flow path
        pyc.connect_flow(self, 'fc.Fl_O', 'inlet.Fl_I')
        pyc.connect_flow(self, 'inlet.Fl_O', 'inlet_duct.Fl_I')
        pyc.connect_flow(self, 'inlet_duct.Fl_O', 'fan.Fl_I')
        pyc.connect_flow(self, 'fan.Fl_O', 'splitter.Fl_I')
        # Core connections
        pyc.connect_flow(self, 'splitter.Fl_O1', 'splitter_core_duct.Fl_I')
        pyc.connect_flow(self, 'splitter_core_duct.Fl_O', 'lpc.Fl_I')
        pyc.connect_flow(self, 'lpc.Fl_O', 'lpc_duct.Fl_I')
        pyc.connect_flow(self, 'lpc_duct.Fl_O', 'hpc.Fl_I')
        pyc.connect_flow(self, 'hpc.Fl_O', 'bld3.Fl_I')
        pyc.connect_flow(self, 'bld3.Fl_O', 'burner.Fl_I')
        pyc.connect_flow(self, 'burner.Fl_O', 'hpt.Fl_I')
        pyc.connect_flow(self, 'hpt.Fl_O', 'hpt_duct.Fl_I')
        pyc.connect_flow(self, 'hpt_duct.Fl_O', 'lpt.Fl_I')
        pyc.connect_flow(self, 'lpt.Fl_O', 'lpt_duct.Fl_I')
        pyc.connect_flow(self, 'lpt_duct.Fl_O','mixer.Fl_I1')
        # Bypass Connections
        pyc.connect_flow(self, 'splitter.Fl_O2', 'bypass_duct.Fl_I')
        pyc.connect_flow(self, 'bypass_duct.Fl_O', 'mixer.Fl_I2')

        #Mixer Connections
        pyc.connect_flow(self, 'mixer.Fl_O', 'mixer_duct.Fl_I')
        # After Burner
        pyc.connect_flow(self,'mixer_duct.Fl_O','afterburner.Fl_I')

        # Nozzle
        pyc.connect_flow(self,'afterburner.Fl_O','mixed_nozz.Fl_I')

        # Connect cooling flows
        pyc.connect_flow(self, 'hpc.cool1', 'lpt.cool1', connect_stat=False)
        pyc.connect_flow(self, 'bld3.cool3', 'hpt.cool3', connect_stat=False)

        # Make additional model connections
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('hpc.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('afterburner.Wfuel', 'perf.Wfuel_1')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('mixed_nozz.Fg', 'perf.Fg_0')

        self.connect('fan.trq', 'lp_shaft.trq_0')
        self.connect('lpc.trq', 'lp_shaft.trq_1')
        self.connect('lpt.trq', 'lp_shaft.trq_2')
        self.connect('hpc.trq', 'hp_shaft.trq_0')
        self.connect('hpt.trq', 'hp_shaft.trq_1')
        self.connect('fc.Fl_O:stat:P', 'mixed_nozz.Ps_exhaust')

        # Add balence components to close the implicit components
        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:
            balance.add_balance('W', lower=1e-3, upper=200., units='lbm/s', eq_units='lbf')
            self.connect('balance.W', 'fc.W')
            self.connect('perf.Fn', 'balance.lhs:W')
            # self.add_subsystem('wDV',IndepVarComp('wDes',100,units='lbm/s'))
            # self.connect('wDV.wDes','fc.W')

            balance.add_balance('BPR', eq_units=None, lower=0.25, val=5.0)
            self.connect('balance.BPR', 'splitter.BPR')
            self.connect('mixer.ER', 'balance.lhs:BPR')

            balance.add_balance('FAR_core', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR_core', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR_core')

            balance.add_balance('FAR_ab', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR_ab', 'afterburner.Fl_I:FAR')
            self.connect('afterburner.Fl_O:tot:T', 'balance.lhs:FAR_ab')

            balance.add_balance('lpt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.lpt_PR', 'lpt.PR')
            self.connect('lp_shaft.pwr_in', 'balance.lhs:lpt_PR')
            self.connect('lp_shaft.pwr_out', 'balance.rhs:lpt_PR')

            balance.add_balance('hpt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.hpt_PR', 'hpt.PR')
            self.connect('hp_shaft.pwr_in', 'balance.lhs:hpt_PR')
            self.connect('hp_shaft.pwr_out', 'balance.rhs:hpt_PR')
        else:

            balance.add_balance('W', lower=1e-3, upper=200., units='lbm/s', eq_units='inch**2')
            self.connect('balance.W', 'fc.W')
            self.connect('mixed_nozz.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('BPR', lower=0.25, upper=5.0, eq_units='psi')
            self.connect('balance.BPR', 'splitter.BPR')
            self.connect('mixer.Fl_I1_calc:stat:P', 'balance.lhs:BPR')
            self.connect('bypass_duct.Fl_O:stat:P', 'balance.rhs:BPR')

            balance.add_balance('FAR_core', eq_units='degR', lower=1e-4, upper=.06, val=.017)
            self.connect('balance.FAR_core', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR_core')

            balance.add_balance('FAR_ab', eq_units='degR', lower=1e-4, upper=.06, val=.017)
            self.connect('balance.FAR_ab', 'afterburner.Fl_I:FAR')
            self.connect('afterburner.Fl_O:tot:T', 'balance.lhs:FAR_ab')

            balance.add_balance('LP_Nmech', val=1., units='rpm', lower=500., eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.LP_Nmech', 'LP_Nmech')
            self.connect('lp_shaft.pwr_in', 'balance.lhs:LP_Nmech')
            self.connect('lp_shaft.pwr_out', 'balance.rhs:LP_Nmech')

            balance.add_balance('HP_Nmech', val=1., units='rpm', lower=500., eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.HP_Nmech', 'HP_Nmech')
            self.connect('hp_shaft.pwr_in', 'balance.lhs:HP_Nmech')
            self.connect('hp_shaft.pwr_out', 'balance.rhs:HP_Nmech')

        # Off design
        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-10
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1


        self.linear_solver = om.DirectSolver(assemble_jac=True)


def print_perf(prob,ptName):
        ''' print out the performancs values'''
        print('BPR',prob[ptName+'.balance.BPR'])
        print('W',prob[ptName+'.balance.W'])
        #print('W',prob[ptName+'.wDV.wDes'])
        print('Fnet uninst.',prob[ptName+'.perf.Fn'])

def page_viewer(point):
    flow_stations = ['fc.Fl_O', 'inlet.Fl_O', 'inlet_duct.Fl_O', 'fan.Fl_O', 'bypass_duct.Fl_O',
                     'splitter.Fl_O2', 'splitter.Fl_O1', 'splitter_core_duct.Fl_O',
                     'lpc.Fl_O', 'lpc_duct.Fl_O', 'hpc.Fl_O', 'bld3.Fl_O', 'burner.Fl_O',
                     'hpt.Fl_O', 'hpt_duct.Fl_O', 'lpt_duct.Fl_O',
                     'mixer.Fl_O', 'mixer_duct.Fl_O', 'afterburner.Fl_O', 'mixed_nozz.Fl_O']

    compressors = ['fan', 'hpc', 'lpc']
    burners = ['burner', 'afterburner']
    turbines = ['hpt', 'lpt']
    shafts = ['hp_shaft', 'lp_shaft']

    print('*'*80)
    print('* ' + ' '*10 + point)
    print('*'*80)
    print_perf(prob, point)

    pyc.print_flow_station(prob,[point+ "."+fl for fl in flow_stations])
    pyc.print_compressor(prob,[point+ "." + c for c in compressors])
    # print_splitter(prob,[point+ ".splitter" ])
    pyc.print_burner(prob,[point+ "." + b for b in burners])
    pyc.print_turbine(prob,[point+ "." + turb for turb in turbines])
    pyc.print_mixer(prob, [point+'.mixer'])
    pyc.print_nozzle(prob, [point + '.mixed_nozz'])
    pyc.print_shaft(prob, [point+ "." + s for s in shafts])
    pyc.print_bleed(prob, [point+'.hpc', point+'.bld3'])

def print_perf(prob,ptName):
    ''' print out the performancs values'''
    print('BPR',prob[ptName+'.balance.BPR'])
    print('W',prob[ptName+'.balance.W'])
    #print('W',prob[ptName+'.wDV.wDes'])
    print('Fnet uninst.',prob[ptName+'.perf.Fn'])

if __name__ == "__main__":
    import time
    from openmdao.api import Problem

    prob = Problem()

    des_vars = prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=["*"])
    element_params = prob.model.add_subsystem('element_params', om.IndepVarComp(), promotes=["*"])

    # FOR DESIGN
    des_vars.add_output('alt', 35000., units='ft') #DV
    des_vars.add_output('MN', 0.8) #DV
    des_vars.add_output('T4max', 3200, units='degR')
    des_vars.add_output('T4_OD', 3100, units='degR')

    des_vars.add_output('T4maxab', 3400, units='degR')
    # des_vars.add_output('FAR_ab', 0)

    des_vars.add_output('Fn_des', 5500.0, units='lbf')
    des_vars.add_output('Mix_ER', 1.05 ,units=None) # defined as 1 over 2
    des_vars.add_output('fan:PRdes', 3.3) #ADV
    des_vars.add_output('lpc:PRdes', 1.935)
    des_vars.add_output('hpc:PRdes', 4.9)


    element_params.add_output('inlet:ram_recovery', 0.9990)
    element_params.add_output('inlet:MN_out', 0.751)

    element_params.add_output('inlet_duct:dPqP', 0.0107)
    element_params.add_output('inlet_duct:MN_out', 0.4463)


    element_params.add_output('fan:effDes', 0.8948)
    element_params.add_output('fan:MN_out', 0.4578)

    #element_params.add_output('splitter:BPR', 5.105) not needed for mixed flow turbofan. balanced based on mixer total pressure ratio
    element_params.add_output('splitter:MN_out1', 0.3104)
    element_params.add_output('splitter:MN_out2', 0.4518)

    element_params.add_output('splitter_core_duct:dPqP', 0.0048)
    element_params.add_output('splitter_core_duct:MN_out', 0.3121)

    element_params.add_output('lpc:effDes', 0.9243)
    element_params.add_output('lpc:MN_out', 0.3059)

    element_params.add_output('lpc_duct:dPqP', 0.0101)
    element_params.add_output('lpc_duct:MN_out', 0.3563)


    element_params.add_output('hpc:effDes', 0.8707)
    element_params.add_output('hpc:MN_out', 0.2442)

    element_params.add_output('bld3:MN_out', 0.3000)

    element_params.add_output('burner:dPqP', 0.0540)
    element_params.add_output('burner:MN_out', 0.1025)

    element_params.add_output('hpt:effDes', 0.8888)
    element_params.add_output('hpt:MN_out', 0.3650)

    element_params.add_output('hpt_duct:dPqP', 0.0051)
    element_params.add_output('hpt_duct:MN_out', 0.3063)

    element_params.add_output('lpt:effDes', 0.8996)
    element_params.add_output('lpt:MN_out', 0.4127)

    element_params.add_output('lpt_duct:dPqP', 0.0107)
    element_params.add_output('lpt_duct:MN_out', 0.4463)

    element_params.add_output('bypass_duct:dPqP', 0.0107)
    element_params.add_output('bypass_duct:MN_out', 0.4463)

    # No params for mixer

    element_params.add_output('mixer_duct:dPqP', 0.0107)
    element_params.add_output('mixer_duct:MN_out', 0.4463)

    element_params.add_output('afterburner:dPqP', 0.0540)
    element_params.add_output('afterburner:MN_out', 0.1025)

    element_params.add_output('mixed_nozz:Cfg', 0.9933)

    element_params.add_output('lp_shaft:Nmech', 4666.1, units='rpm')
    element_params.add_output('hp_shaft:Nmech', 14705.7, units='rpm')
    # element_params.add_output('hp_shaft:HPX', 250.0, units='hp')
    element_params.add_output('hp_shaft:HPX', 0.0, units='hp')

    element_params.add_output('hpc:cool1:frac_W', 0.050708)
    element_params.add_output('hpc:cool1:frac_P', 0.5)
    element_params.add_output('hpc:cool1:frac_work', 0.5)

    element_params.add_output('bld3:cool3:frac_W', 0.067214)

    element_params.add_output('hpt:cool3:frac_P', 1.0)
    element_params.add_output('lpt:cool1:frac_P', 1.0)

    #####################
    # DESIGN CASE
    #####################

    prob.model.add_subsystem('DESIGN', MixedFlowTurbofan(design=True))

    prob.model.connect('alt', 'DESIGN.fc.alt')
    prob.model.connect('MN', 'DESIGN.fc.MN')
    prob.model.connect('Fn_des', 'DESIGN.balance.rhs:W')
    prob.model.connect('T4max', 'DESIGN.balance.rhs:FAR_core')

    prob.model.connect('T4maxab', 'DESIGN.balance.rhs:FAR_ab')
    # prob.model.connect('FAR_ab', 'DESIGN.afterburner.Fl_I:FAR')


    prob.model.connect('Mix_ER', 'DESIGN.balance.rhs:BPR')

    prob.model.connect('inlet:ram_recovery', 'DESIGN.inlet.ram_recovery')
    prob.model.connect('inlet:MN_out', 'DESIGN.inlet.MN')

    prob.model.connect('inlet_duct:dPqP', 'DESIGN.inlet_duct.dPqP')
    prob.model.connect('inlet_duct:MN_out', 'DESIGN.inlet_duct.MN')

    prob.model.connect('fan:PRdes', 'DESIGN.fan.PR')
    prob.model.connect('fan:effDes', 'DESIGN.fan.eff')
    prob.model.connect('fan:MN_out', 'DESIGN.fan.MN')

    #prob.model.connect('splitter:BPR', 'DESIGN.splitter.BPR')
    prob.model.connect('splitter:MN_out1', 'DESIGN.splitter.MN1')
    prob.model.connect('splitter:MN_out2', 'DESIGN.splitter.MN2')

    prob.model.connect('splitter_core_duct:dPqP', 'DESIGN.splitter_core_duct.dPqP')
    prob.model.connect('splitter_core_duct:MN_out', 'DESIGN.splitter_core_duct.MN')

    prob.model.connect('lpc:PRdes', 'DESIGN.lpc.PR')
    prob.model.connect('lpc:effDes', 'DESIGN.lpc.eff')
    prob.model.connect('lpc:MN_out', 'DESIGN.lpc.MN')

    prob.model.connect('lpc_duct:dPqP', 'DESIGN.lpc_duct.dPqP')
    prob.model.connect('lpc_duct:MN_out', 'DESIGN.lpc_duct.MN')

    prob.model.connect('hpc:PRdes', 'DESIGN.hpc.PR')
    prob.model.connect('hpc:effDes', 'DESIGN.hpc.eff')
    prob.model.connect('hpc:MN_out', 'DESIGN.hpc.MN')

    prob.model.connect('bld3:MN_out', 'DESIGN.bld3.MN')

    prob.model.connect('burner:dPqP', 'DESIGN.burner.dPqP')
    prob.model.connect('burner:MN_out', 'DESIGN.burner.MN')

    prob.model.connect('hpt:effDes', 'DESIGN.hpt.eff')
    prob.model.connect('hpt:MN_out', 'DESIGN.hpt.MN')

    prob.model.connect('hpt_duct:dPqP', 'DESIGN.hpt_duct.dPqP')
    prob.model.connect('hpt_duct:MN_out', 'DESIGN.hpt_duct.MN')

    prob.model.connect('lpt:effDes', 'DESIGN.lpt.eff')
    prob.model.connect('lpt:MN_out', 'DESIGN.lpt.MN')

    prob.model.connect('lpt_duct:dPqP', 'DESIGN.lpt_duct.dPqP')
    prob.model.connect('lpt_duct:MN_out', 'DESIGN.lpt_duct.MN')

    prob.model.connect('bypass_duct:dPqP', 'DESIGN.bypass_duct.dPqP')
    prob.model.connect('bypass_duct:MN_out', 'DESIGN.bypass_duct.MN')

    prob.model.connect('mixer_duct:dPqP', 'DESIGN.mixer_duct.dPqP')
    prob.model.connect('mixer_duct:MN_out', 'DESIGN.mixer_duct.MN')

    prob.model.connect('afterburner:dPqP', 'DESIGN.afterburner.dPqP')
    prob.model.connect('afterburner:MN_out', 'DESIGN.afterburner.MN')

    prob.model.connect('mixed_nozz:Cfg', 'DESIGN.mixed_nozz.Cfg')

    prob.model.connect('lp_shaft:Nmech', 'DESIGN.LP_Nmech')
    prob.model.connect('hp_shaft:Nmech', 'DESIGN.HP_Nmech')
    prob.model.connect('hp_shaft:HPX', 'DESIGN.hp_shaft.HPX')

    prob.model.connect('hpc:cool1:frac_W', 'DESIGN.hpc.cool1:frac_W')
    prob.model.connect('hpc:cool1:frac_P', 'DESIGN.hpc.cool1:frac_P')
    prob.model.connect('hpc:cool1:frac_work', 'DESIGN.hpc.cool1:frac_work')

    prob.model.connect('bld3:cool3:frac_W', 'DESIGN.bld3.cool3:frac_W')

    prob.model.connect('hpt:cool3:frac_P', 'DESIGN.hpt.cool3:frac_P')
    prob.model.connect('lpt:cool1:frac_P', 'DESIGN.lpt.cool1:frac_P')


    ####################
    # OFF DESIGN CASES
    ####################
    od_pts = ['OD0',]
    # od_pts = []

    od_alts = [35000,]
    od_MNs = [0.8, ]

    des_vars.add_output('OD:alts', val=od_alts, units='ft')
    des_vars.add_output('OD:MNs', val=od_MNs)


    for i,pt in enumerate(od_pts):
        prob.model.add_subsystem(pt, MixedFlowTurbofan(design=False))

        prob.model.connect('OD:alts', pt+'.fc.alt', src_indices=[i,])
        prob.model.connect('OD:MNs', pt+'.fc.MN', src_indices=[i,])

        prob.model.connect('T4_OD', pt+'.balance.rhs:FAR_core')
        prob.model.connect('T4maxab', pt+'.balance.rhs:FAR_ab')
        # prob.model.connect('FAR_ab', pt+'.afterburner.Fl_I:FAR')

        prob.model.connect('inlet:ram_recovery', pt+'.inlet.ram_recovery')
        prob.model.connect('mixed_nozz:Cfg', pt+'.mixed_nozz.Cfg')
        prob.model.connect('hp_shaft:HPX', pt+'.hp_shaft.HPX')


        # duct pressure losses
        prob.model.connect('inlet_duct:dPqP', pt+'.inlet_duct.dPqP')
        prob.model.connect('splitter_core_duct:dPqP', pt+'.splitter_core_duct.dPqP')
        prob.model.connect('bypass_duct:dPqP', pt+'.bypass_duct.dPqP')
        prob.model.connect('lpc_duct:dPqP', pt+'.lpc_duct.dPqP')
        prob.model.connect('hpt_duct:dPqP', pt+'.hpt_duct.dPqP')
        prob.model.connect('lpt_duct:dPqP', pt+'.lpt_duct.dPqP')
        prob.model.connect('mixer_duct:dPqP', pt+'.mixer_duct.dPqP')

        # burner pressure losses
        prob.model.connect('burner:dPqP', pt+'.burner.dPqP')
        prob.model.connect('afterburner:dPqP', pt+'.afterburner.dPqP')

        # cooling flow fractions
        prob.model.connect('hpc:cool1:frac_W', pt+'.hpc.cool1:frac_W')
        prob.model.connect('hpc:cool1:frac_P', pt+'.hpc.cool1:frac_P')
        prob.model.connect('hpc:cool1:frac_work', pt+'.hpc.cool1:frac_work')
        prob.model.connect('bld3:cool3:frac_W', pt+'.bld3.cool3:frac_W')
        prob.model.connect('hpt:cool3:frac_P', pt+'.hpt.cool3:frac_P')
        prob.model.connect('lpt:cool1:frac_P', pt+'.lpt.cool1:frac_P')

        # map scalars
        prob.model.connect('DESIGN.fan.s_PR', pt+'.fan.s_PR')
        prob.model.connect('DESIGN.fan.s_Wc', pt+'.fan.s_Wc')
        prob.model.connect('DESIGN.fan.s_eff', pt+'.fan.s_eff')
        prob.model.connect('DESIGN.fan.s_Nc', pt+'.fan.s_Nc')
        prob.model.connect('DESIGN.lpc.s_PR', pt+'.lpc.s_PR')
        prob.model.connect('DESIGN.lpc.s_Wc', pt+'.lpc.s_Wc')
        prob.model.connect('DESIGN.lpc.s_eff', pt+'.lpc.s_eff')
        prob.model.connect('DESIGN.lpc.s_Nc', pt+'.lpc.s_Nc')
        prob.model.connect('DESIGN.hpc.s_PR', pt+'.hpc.s_PR')
        prob.model.connect('DESIGN.hpc.s_Wc', pt+'.hpc.s_Wc')
        prob.model.connect('DESIGN.hpc.s_eff', pt+'.hpc.s_eff')
        prob.model.connect('DESIGN.hpc.s_Nc', pt+'.hpc.s_Nc')
        prob.model.connect('DESIGN.hpt.s_PR', pt+'.hpt.s_PR')
        prob.model.connect('DESIGN.hpt.s_Wp', pt+'.hpt.s_Wp')
        prob.model.connect('DESIGN.hpt.s_eff', pt+'.hpt.s_eff')
        prob.model.connect('DESIGN.hpt.s_Np', pt+'.hpt.s_Np')
        prob.model.connect('DESIGN.lpt.s_PR', pt+'.lpt.s_PR')
        prob.model.connect('DESIGN.lpt.s_Wp', pt+'.lpt.s_Wp')
        prob.model.connect('DESIGN.lpt.s_eff', pt+'.lpt.s_eff')
        prob.model.connect('DESIGN.lpt.s_Np', pt+'.lpt.s_Np')

        # flow areas
        prob.model.connect('DESIGN.mixed_nozz.Throat:stat:area', pt+'.balance.rhs:W')

        prob.model.connect('DESIGN.inlet.Fl_O:stat:area', pt+'.inlet.area')
        prob.model.connect('DESIGN.fan.Fl_O:stat:area', pt+'.fan.area')
        prob.model.connect('DESIGN.splitter.Fl_O1:stat:area', pt+'.splitter.area1')
        prob.model.connect('DESIGN.splitter.Fl_O2:stat:area', pt+'.splitter.area2')
        prob.model.connect('DESIGN.splitter_core_duct.Fl_O:stat:area', pt+'.splitter_core_duct.area')
        prob.model.connect('DESIGN.lpc.Fl_O:stat:area', pt+'.lpc.area')
        prob.model.connect('DESIGN.lpc_duct.Fl_O:stat:area', pt+'.lpc_duct.area')
        prob.model.connect('DESIGN.hpc.Fl_O:stat:area', pt+'.hpc.area')
        prob.model.connect('DESIGN.bld3.Fl_O:stat:area', pt+'.bld3.area')
        prob.model.connect('DESIGN.burner.Fl_O:stat:area', pt+'.burner.area')
        prob.model.connect('DESIGN.hpt.Fl_O:stat:area', pt+'.hpt.area')
        prob.model.connect('DESIGN.hpt_duct.Fl_O:stat:area', pt+'.hpt_duct.area')
        prob.model.connect('DESIGN.lpt.Fl_O:stat:area', pt+'.lpt.area')
        prob.model.connect('DESIGN.lpt_duct.Fl_O:stat:area', pt+'.lpt_duct.area')
        prob.model.connect('DESIGN.bypass_duct.Fl_O:stat:area', pt+'.bypass_duct.area')
        prob.model.connect('DESIGN.mixer.Fl_O:stat:area', pt+'.mixer.area')
        prob.model.connect('DESIGN.mixer.Fl_I1_calc:stat:area', pt+'.mixer.Fl_I1_stat_calc.area')
        prob.model.connect('DESIGN.mixer_duct.Fl_O:stat:area', pt+'.mixer_duct.area')
        prob.model.connect('DESIGN.afterburner.Fl_O:stat:area', pt+'.afterburner.area')


    # setup problem
    prob.setup(check=False)#True)

    # initial guesses
    prob['DESIGN.balance.FAR_core'] = 0.025
    prob['DESIGN.balance.FAR_ab'] = 0.025
    prob['DESIGN.balance.BPR'] = 1.0
    prob['DESIGN.balance.W'] = 100.
    prob['DESIGN.balance.lpt_PR'] = 3.5
    prob['DESIGN.balance.hpt_PR'] = 2.5
    prob['DESIGN.fc.balance.Pt'] = 5.2
    prob['DESIGN.fc.balance.Tt'] = 440.0
    prob['DESIGN.mixer.balance.P_tot']= 15

    for pt in od_pts:
        prob[pt+'.balance.FAR_core'] = 0.025
        prob[pt+'.balance.FAR_ab'] = 0.025
        prob[pt+'.balance.BPR'] = 2.5
        prob[pt+'.balance.W'] = 50.
        prob[pt+'.balance.HP_Nmech'] = 14000
        prob[pt+'.balance.LP_Nmech'] = 4000
        prob[pt+'.fc.balance.Pt'] = 5.2
        prob[pt+'.fc.balance.Tt'] = 440.0
        prob[pt+'.mixer.balance.P_tot']= 15
        prob[pt+'.hpt.PR'] = 2.0
        prob[pt+'.lpt.PR'] = 4.0
        prob[pt+'.fan.map.RlineMap'] = 2.0
        prob[pt+'.lpc.map.RlineMap'] = 2.0
        prob[pt+'.hpc.map.RlineMap'] = 2.0

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)


    prob.run_model()
    page_viewer('DESIGN')


    for T in [3200, 3100, 3000]:
        prob['T4_OD'] = T

        prob.run_model()

        page_viewer('OD0')

    print()
    print("time", time.time() - st)

