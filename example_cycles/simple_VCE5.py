import sys

import numpy as np

import openmdao.api as om

import pycycle.api as pyc

from FAR_Balance import FAR_Balance


class sVCE5(pyc.Cycle):

    def initialize(self):
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')
        self.options.declare('cooling', default=True, 
                             desc='If True, calculate cooling flows')

    def setup(self):

        thermo_spec = pyc.species_data.janaf
        design = self.options['design']
        cooling = self.options['cooling']
        
        self.pyc_add_element('fc', pyc.FlightConditions(thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.pyc_add_element('inlet', pyc.ComplexInlet(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.pyc_add_element('duct1', pyc.Duct(design=design, expMN=1.75, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.pyc_add_element('fan', pyc.Compressor(map_data=pyc.FanMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX, 
                                                 map_extrap=True), promotes_inputs=[('Nmech','LP_Nmech')])
        self.pyc_add_element('duct21', pyc.Duct(design=design, expMN=1.75, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        
        self.pyc_add_element('splitter', pyc.Splitter(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.pyc_add_element('ductbypassa', pyc.Duct(design=design, expMN=1.75, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.pyc_add_element('ductcore', pyc.Duct(design=design, expMN=1.75, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        
        #ENGINE CORE
        self.pyc_add_element('hpc', pyc.Compressor(map_data=pyc.HPCMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX,
                                        bleed_names=['bleed_hpc42','bleed_hpc49','bleed_hpc1', 'bleed_hpc2'], map_extrap=True),promotes_inputs=[('Nmech','HP_Nmech')])
        self.pyc_add_element('bleeds', pyc.BleedOut(design=design, bleed_names=['cust','bleed_srce1', 'bleed_srce2']))
        self.pyc_add_element('duct3', pyc.Duct(design=design, expMN=1.75, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.pyc_add_element('combustor', pyc.Combustor(design=design,thermo_data=thermo_spec,
                                        inflow_elements=pyc.AIR_MIX,
                                        air_fuel_elements=pyc.AIR_FUEL_MIX,
                                        fuel_type='Jet-A(g)'))
        self.pyc_add_element('duct40', pyc.Duct(design=design, expMN=1.75, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.pyc_add_element('hpt', pyc.Turbine(map_data=pyc.HPTMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX,
                                        bleed_names=['bleed_srce1','bleed_srce2', 'bleed_42'], map_extrap=True),promotes_inputs=[('Nmech','HP_Nmech')])
        self.pyc_add_element('duct42', pyc.Duct(design=design, expMN=1.75, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.pyc_add_element('lpt', pyc.Turbine(map_data=pyc.LPTMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX,
                                        bleed_names=['bleed_lpt1','bleed_lpt2', 'bleed_49'], map_extrap=True),promotes_inputs=[('Nmech','LP_Nmech')])
        
        #MIXER MERGE
        self.pyc_add_element('duct7', pyc.Duct(design=design, expMN=1.75, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.pyc_add_element('ductbypassb', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.pyc_add_element('mixer', pyc.Mixer(design=design, thermo_data=thermo_spec, designed_stream=1, 
                                          Fl_I1_elements=pyc.AIR_FUEL_MIX, Fl_I2_elements=pyc.AIR_MIX))
        
        #AFTERBURNER
        self.pyc_add_element('duct8', pyc.Duct(design=design, expMN=1.75, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.pyc_add_element('afterburner', pyc.Combustor(design=design, thermo_data=thermo_spec,inflow_elements=pyc.AIR_FUEL_MIX,
                                            air_fuel_elements=pyc.AIR_FUEL_MIX, fuel_type='Jet-A(g)'))
        self.pyc_add_element('nozzle', pyc.Nozzle(nozzType='CD', lossCoef='Cv', thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        
        self.pyc_add_element('lp_shaft', pyc.Shaft(num_ports=2),promotes_inputs=[('Nmech','LP_Nmech')])
        self.pyc_add_element('hp_shaft', pyc.Shaft(num_ports=2),promotes_inputs=[('Nmech','HP_Nmech')])
        
        add_order = []
        
        if not design:
            self.add_subsystem('FAR_bal', FAR_Balance())
        
            self.connect('combustor.Fl_O:tot:T', 'FAR_bal.T4')
            self.connect('fan.map.NcMap', 'FAR_bal.NcMapVal')
        
            self.connect('FAR_bal.FAR', 'combustor.Fl_I:FAR')
            
            add_order = add_order + ['FAR_bal']
        
        #PERFIRMANCE MODULE
        self.pyc_add_element('perf', pyc.Performance(num_nozzles=1, num_burners=2))

        #OPR
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('hpc.Fl_O:tot:P', 'perf.Pt3')
        
        # FUEL BURN
        self.connect('combustor.Wfuel', 'perf.Wfuel_0')
        self.connect('afterburner.Wfuel', 'perf.Wfuel_1')
        
        #Net Force
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozzle.Fg', 'perf.Fg_0')

        self.connect('fan.trq', 'lp_shaft.trq_0')
        self.connect('lpt.trq', 'lp_shaft.trq_1')
        self.connect('hpc.trq', 'hp_shaft.trq_0')
        self.connect('hpt.trq', 'hp_shaft.trq_1')
        self.connect('fc.Fl_O:stat:P', 'nozzle.Ps_exhaust')

        #
        #FLOW STATION CONNECT
        #
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O', 'duct1.Fl_I')
       
        self.pyc_connect_flow('duct1.Fl_O', 'fan.Fl_I')
        self.pyc_connect_flow('fan.Fl_O', 'duct21.Fl_I')
        self.pyc_connect_flow('duct21.Fl_O', 'splitter.Fl_I')
        self.pyc_connect_flow('splitter.Fl_O1', 'ductcore.Fl_I')
        self.pyc_connect_flow('splitter.Fl_O2', 'ductbypassa.Fl_I')
        
        self.pyc_connect_flow('ductcore.Fl_O', 'hpc.Fl_I')
        self.pyc_connect_flow('hpc.Fl_O', 'bleeds.Fl_I')
        self.pyc_connect_flow('bleeds.Fl_O', 'duct3.Fl_I')
        self.pyc_connect_flow('duct3.Fl_O', 'combustor.Fl_I')
        self.pyc_connect_flow('combustor.Fl_O', 'duct40.Fl_I') 
        self.pyc_connect_flow('duct40.Fl_O', 'hpt.Fl_I')
        self.pyc_connect_flow('hpt.Fl_O', 'duct42.Fl_I')
        self.pyc_connect_flow('duct42.Fl_O', 'lpt.Fl_I')
        self.pyc_connect_flow('lpt.Fl_O', 'duct7.Fl_I')

        self.pyc_connect_flow('ductbypassa.Fl_O', 'ductbypassb.Fl_I')
        
        self.pyc_connect_flow('ductbypassb.Fl_O', 'mixer.Fl_I2')
        self.pyc_connect_flow('duct7.Fl_O', 'mixer.Fl_I1')
        
        self.pyc_connect_flow('mixer.Fl_O', 'duct8.Fl_I')
        self.pyc_connect_flow('duct8.Fl_O', 'afterburner.Fl_I')
        self.pyc_connect_flow('afterburner.Fl_O', 'nozzle.Fl_I')
                
        #
        #Cooling flow connections
        #
        self.pyc_connect_flow('hpc.bleed_hpc1', 'lpt.bleed_lpt1', connect_stat=False)
        self.pyc_connect_flow('hpc.bleed_hpc2', 'lpt.bleed_lpt2', connect_stat=False)
        self.pyc_connect_flow('hpc.bleed_hpc49', 'lpt.bleed_49', connect_stat=False)
        self.pyc_connect_flow('hpc.bleed_hpc42', 'hpt.bleed_42', connect_stat=False)
        self.pyc_connect_flow('bleeds.bleed_srce1', 'hpt.bleed_srce1', connect_stat=False)
        self.pyc_connect_flow('bleeds.bleed_srce2', 'hpt.bleed_srce2', connect_stat=False)

        
        
        
        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:
            
            balance.add_balance('BPR', eq_units=None, lower=0.1, val=4)
            self.connect('balance.BPR', 'splitter.BPR')
            self.connect('mixer.ER', 'balance.lhs:BPR') 
            
            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.032)
            self.connect('balance.FAR', 'combustor.Fl_I:FAR')
            self.connect('combustor.Fl_O:tot:T', 'balance.lhs:FAR')
                      
            balance.add_balance('lpt_PR', val=1.5, lower=1.001, upper=8,
                                eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.lpt_PR', 'lpt.PR')
            self.connect('lp_shaft.pwr_in_real', 'balance.lhs:lpt_PR')
            self.connect('lp_shaft.pwr_out_real', 'balance.rhs:lpt_PR')

            balance.add_balance('hpt_PR', val=1.5, lower=1.001, upper=8,
                                eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.hpt_PR', 'hpt.PR')
            self.connect('hp_shaft.pwr_in_real', 'balance.lhs:hpt_PR')
            self.connect('hp_shaft.pwr_out_real', 'balance.rhs:hpt_PR')
            
            # 
            # Efficiencies
            # 
            balance.add_balance('fan_eff', val=0.9689, lower=0.01, upper=1.0, units=None, eq_units=None)
            self.connect('balance.fan_eff', 'fan.eff')
            self.connect('fan.eff_poly', 'balance.lhs:fan_eff')

            balance.add_balance('hpc_eff', val=0.8470, lower=0.01, upper=1.0, units=None, eq_units=None)
            self.connect('balance.hpc_eff', 'hpc.eff')
            self.connect('hpc.eff_poly', 'balance.lhs:hpc_eff')

            balance.add_balance('hpt_eff', val=0.9226, lower=0.01, upper=1.0, units=None, eq_units=None)
            self.connect('balance.hpt_eff', 'hpt.eff')
            self.connect('hpt.eff_poly', 'balance.lhs:hpt_eff')

            balance.add_balance('lpt_eff', val=0.9401, lower=0.01, upper=1.0, units=None, eq_units=None)
            self.connect('balance.lpt_eff', 'lpt.eff')
            self.connect('lpt.eff_poly', 'balance.lhs:lpt_eff')

        else:

            balance.add_balance('W', units='lbm/s', lower=10., upper=1000., rhs_val=25)
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('fan.SMW', 'balance.lhs:W')

            balance.add_balance('BPR', lower=0.3, eq_units='psi')
            self.connect('balance.BPR', 'splitter.BPR')
            self.connect('ductbypassb.Fl_O:stat:P', 'balance.rhs:BPR')
            self.connect('mixer.Fl_I1_calc:stat:P', 'balance.lhs:BPR')
            
            balance.add_balance('lp_Nmech', val=1.5, units='rpm', lower=500., eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.lp_Nmech', 'LP_Nmech')
            self.connect('lp_shaft.pwr_in_real', 'balance.lhs:lp_Nmech')
            self.connect('lp_shaft.pwr_out_real', 'balance.rhs:lp_Nmech')

            balance.add_balance('hp_Nmech', val=1.5, units='rpm', lower=500., eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.hp_Nmech', 'HP_Nmech')
            self.connect('hp_shaft.pwr_in_real', 'balance.lhs:hp_Nmech')
            self.connect('hp_shaft.pwr_out_real', 'balance.rhs:hp_Nmech')



        if cooling:
            #
            # HIGH PRESSURE TURBINE COOLING
            #
            self.add_subsystem('hpt_cooling', pyc.TurbineCooling(n_stages=2, thermo_data=pyc.species_data.janaf, owns_x_factor=True, T_metal=2350.))
            self.add_subsystem('hpt_chargable', pyc.CombineCooling(n_ins=3))
            
            self.pyc_connect_flow('bleeds.bleed_srce1', 'hpt_cooling.Fl_cool', connect_stat=False)
            self.pyc_connect_flow('combustor.Fl_O', 'hpt_cooling.Fl_turb_I')
            self.pyc_connect_flow('hpt.Fl_O', 'hpt_cooling.Fl_turb_O')
            
            self.connect('hpt_cooling.row_1.W_cool', 'hpt_chargable.W_1')
            self.connect('hpt_cooling.row_2.W_cool', 'hpt_chargable.W_2')
            self.connect('hpt_cooling.row_3.W_cool', 'hpt_chargable.W_3')
            self.connect('hpt.power', 'hpt_cooling.turb_pwr')
            
            balance.add_balance('hpt_nochrg_cool_frac', val=0.06366, lower=0.01, upper=0.2, eq_units='lbm/s')
            self.connect('balance.hpt_nochrg_cool_frac', 'bleeds.bleed_srce1:frac_W')
            self.connect('bleeds.bleed_srce1:stat:W', 'balance.lhs:hpt_nochrg_cool_frac')
            self.connect('hpt_cooling.row_0.W_cool', 'balance.rhs:hpt_nochrg_cool_frac')
            
            balance.add_balance('hpt_chrg_cool_frac', val=0.07037, lower=0.01, upper=0.2, eq_units='lbm/s')
            self.connect('balance.hpt_chrg_cool_frac', 'bleeds.bleed_srce2:frac_W')
            self.connect('bleeds.bleed_srce2:stat:W', 'balance.lhs:hpt_chrg_cool_frac')
            self.connect('hpt_chargable.W_cool', 'balance.rhs:hpt_chrg_cool_frac')
            
            #
            # LOW PRESSURE TURBINE COOLING
            #
            self.add_subsystem('lpt_cooling', pyc.TurbineCooling(n_stages=2, thermo_data=pyc.species_data.janaf, owns_x_factor=True, T_metal=2350.))
            self.add_subsystem('lpt_chargable', pyc.CombineCooling(n_ins=3))
            
            self.pyc_connect_flow('hpc.bleed_hpc1', 'lpt_cooling.Fl_cool', connect_stat=False)
            self.pyc_connect_flow('duct42.Fl_O', 'lpt_cooling.Fl_turb_I')
            self.pyc_connect_flow('lpt.Fl_O', 'lpt_cooling.Fl_turb_O')
            
            self.connect('lpt_cooling.row_1.W_cool', 'lpt_chargable.W_1')
            self.connect('lpt_cooling.row_2.W_cool', 'lpt_chargable.W_2')
            self.connect('lpt_cooling.row_3.W_cool', 'lpt_chargable.W_3')
            self.connect('lpt.power', 'lpt_cooling.turb_pwr')
            
            balance.add_balance('lpt_nochrg_cool_frac', val=0.06366, lower=0.00001, upper=0.2, eq_units='lbm/s')
            self.connect('balance.lpt_nochrg_cool_frac', 'hpc.bleed_hpc1:frac_W')
            self.connect('hpc.bleed_hpc1:stat:W', 'balance.lhs:lpt_nochrg_cool_frac')
            self.connect('lpt_cooling.row_0.W_cool', 'balance.rhs:lpt_nochrg_cool_frac')
            
            balance.add_balance('lpt_chrg_cool_frac', val=0.07037, lower=0.00001, upper=0.2, eq_units='lbm/s')
            self.connect('balance.lpt_chrg_cool_frac', 'hpc.bleed_hpc2:frac_W')
            self.connect('hpc.bleed_hpc2:stat:W', 'balance.lhs:lpt_chrg_cool_frac')
            self.connect('lpt_chargable.W_cool', 'balance.rhs:lpt_chrg_cool_frac')
            
            add_order = add_order + ['hpt_cooling', 'hpt_chargable', 'lpt_cooling', 'lpt_chargable']
        
        
        main_order = ['balance', 'fc', 'inlet', 'duct1', 'fan', 'duct21', 'splitter', 'ductbypassa', 'ductcore', 'hpc',
                        'bleeds', 'duct3', 'combustor', 'duct40', 'hpt', 'duct42', 'lpt', 'duct7', 
                        'ductbypassb', 'mixer', 'duct8', 'afterburner', 'nozzle', 'lp_shaft', 'hp_shaft','perf']
        
        self.set_order(main_order + add_order)
        
        
        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-8
        newton.options['rtol'] = 1e-8
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 35
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False
        # ls = newton.linesearch = BoundsEnforceLS()
        ls = newton.linesearch = om.ArmijoGoldsteinLS()
        ls.options['maxiter'] = 5
        ls.options['bound_enforcement'] = 'scalar'
        #ls.options['iprint'] = -1

        self.linear_solver = om.DirectSolver(assemble_jac=True)

def viewer(prob, pt, file=sys.stdout):
    """
    print a report of all the relevant cycle properties
    """

    if pt == 'DESIGN':
        MN = prob['DESIGN.fc.Fl_O:stat:MN']
        LPT_PR = prob['DESIGN.balance.lpt_PR']
        HPT_PR = prob['DESIGN.balance.hpt_PR']
        FAR = prob['DESIGN.balance.FAR']
    else:
        MN = prob[pt+'.fc.Fl_O:stat:MN']
        LPT_PR = prob[pt+'.lpt.PR']
        HPT_PR = prob[pt+'.hpt.PR']
        FAR = prob[pt+'.FAR_bal.FAR']

    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC      BPR ", file=file, flush=True)
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f  %7.3f" %(MN, prob[pt+'.fc.alt'],prob[pt+'.inlet.Fl_O:stat:W'],prob[pt+'.perf.Fn'],prob[pt+'.perf.Fg'],prob[pt+'.inlet.F_ram'],prob[pt+'.perf.OPR'],prob[pt+'.perf.TSFC'], prob[pt+'.splitter.BPR']), file=file, flush=True)


    fs_names = ['fc.Fl_O', 'inlet.Fl_O', 'duct1.Fl_O', 'fan.Fl_O', 'duct21.Fl_O', 'splitter.Fl_O1', 'splitter.Fl_O2',
                'ductbypassa.Fl_O', 'ductcore.Fl_O','hpc.Fl_O', 'bleeds.Fl_O', 'duct3.Fl_O', 'combustor.Fl_O','duct40.Fl_O', 'hpt.Fl_O',
                'duct42.Fl_O', 'lpt.Fl_O', 'duct7.Fl_O', 'ductbypassb.Fl_O', 'mixer.Fl_O', 'duct8.Fl_O', 'afterburner.Fl_O', 'nozzle.Fl_O']
    
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    comp_names = ['fan', 'hpc']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.print_compressor(prob, comp_full_names, file=file)

    burn_names = ['combustor', 'afterburner']
    burn_full_names = [f'{pt}.{b}' for b in burn_names]
    pyc.print_burner(prob, burn_full_names, file=file)

    turb_names = ['hpt', 'lpt']
    turb_full_names = [f'{pt}.{t}' for t in turb_names]
    pyc.print_turbine(prob, turb_full_names, file=file)

    noz_names = ['nozzle']
    noz_full_names = [f'{pt}.{n}' for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)
    
    pyc.print_mixer(prob, [pt+'.mixer'])

    shaft_names = ['hp_shaft', 'lp_shaft']
    shaft_full_names = [f'{pt}.{s}' for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=file)

    bleed_names = ['hpc', 'bleeds']
    bleed_full_names = [f'{pt}.{b}' for b in bleed_names]
    pyc.print_bleed(prob, bleed_full_names, file=file)  

class MPsimple_VCE5(pyc.MPCycle):
        
    def setup(self):
        
        #Design point
        DESIGN = self.pyc_add_pnt('DESIGN', sVCE5(design=True, cooling=False))
        
        designcooling = DESIGN.options['cooling']

        self.set_input_defaults('DESIGN.fan.PR', 2.8)
        self.set_input_defaults('DESIGN.hpc.PR', 14.85)        
        self.set_input_defaults('DESIGN.fc.alt', 0.1, units='ft')
        self.set_input_defaults('DESIGN.fc.MN', 0.2)
        self.set_input_defaults('DESIGN.balance.rhs:FAR', 3660, units='degR') #Maximum Burner Temperature
        self.set_input_defaults('DESIGN.balance.rhs:BPR', 1.01, units=None) #Mixer Extraction Ratio
        self.set_input_defaults('DESIGN.inlet.Fl_I:stat:W', 570, units='lbm/s')
        
        self.set_input_defaults('DESIGN.inlet.MN', 0.45)
        self.set_input_defaults('DESIGN.duct1.MN', 0.45)
        self.set_input_defaults('DESIGN.fan.MN', 0.3)
        self.set_input_defaults('DESIGN.duct21.MN', 0.25)
        self.set_input_defaults('DESIGN.splitter.MN1', 0.2)
        self.set_input_defaults('DESIGN.splitter.MN2', 0.4)
        self.set_input_defaults('DESIGN.ductcore.MN', 0.2)  
        self.set_input_defaults('DESIGN.ductbypassa.MN', 0.2)    
        self.set_input_defaults('DESIGN.hpc.MN', 0.4)
        self.set_input_defaults('DESIGN.bleeds.MN', 0.15)
        self.set_input_defaults('DESIGN.duct3.MN', 0.0978)       
        self.set_input_defaults('DESIGN.combustor.MN', 0.15)
        self.set_input_defaults('DESIGN.duct40.MN', 0.1)
        self.set_input_defaults('DESIGN.hpt.MN', 0.4)
        self.set_input_defaults('DESIGN.duct42.MN', 0.4)
        self.set_input_defaults('DESIGN.lpt.MN', 3.0)
        self.set_input_defaults('DESIGN.duct7.MN', 0.2)
        self.set_input_defaults('DESIGN.ductbypassb.MN', 0.2)
        self.set_input_defaults('DESIGN.duct8.MN', 0.25)
        self.set_input_defaults('DESIGN.afterburner.MN', 0.2573)
        self.set_input_defaults('DESIGN.afterburner.Fl_I:FAR', 0.00001)
        
        self.set_input_defaults('DESIGN.balance.rhs:fan_eff', 0.915) #Polytropic efficiency
        self.set_input_defaults('DESIGN.balance.rhs:hpc_eff', 0.915) #Polytropic efficiency
        self.set_input_defaults('DESIGN.balance.rhs:hpt_eff', 0.890) #Polytropic efficiency
        self.set_input_defaults('DESIGN.balance.rhs:lpt_eff', 0.905) #Polytropic efficiency
    
        self.set_input_defaults('DESIGN.LP_Nmech', 3000, units='rpm')
        self.set_input_defaults('DESIGN.HP_Nmech', 6000, units='rpm')
        
        
        if not designcooling:
            self.set_input_defaults('DESIGN.hpc.bleed_hpc1:frac_W', 0.0001)
            self.set_input_defaults('DESIGN.hpc.bleed_hpc2:frac_W', 0.0001)
            self.set_input_defaults('DESIGN.bleeds.bleed_srce1:frac_W', 0.01)
            self.set_input_defaults('DESIGN.bleeds.bleed_srce2:frac_W', 0.01)
            
        #Set Cycle Parameters (those true troughout all scenarios)
        self.pyc_add_cycle_param('inlet.etaBase', 0.998)
        self.pyc_add_cycle_param('duct1.dPqP', 0.0)
        self.pyc_add_cycle_param('duct21.dPqP', 0.005)
        self.pyc_add_cycle_param('ductcore.dPqP', 0.025)
        self.pyc_add_cycle_param('ductbypassa.dPqP', 0.008)
        self.pyc_add_cycle_param('duct3.dPqP', 0.005)
        self.pyc_add_cycle_param('combustor.dPqP', 0.035)
        self.pyc_add_cycle_param('duct40.dPqP', 0.005)
        self.pyc_add_cycle_param('duct42.dPqP', 0.005)
        self.pyc_add_cycle_param('duct7.dPqP', 0.005)
        self.pyc_add_cycle_param('ductbypassb.dPqP', 0.022)
        self.pyc_add_cycle_param('duct8.dPqP', 0.0)
        self.pyc_add_cycle_param('afterburner.dPqP', 0.0265)
        self.pyc_add_cycle_param('nozzle.Cv', 0.978)
        self.pyc_add_cycle_param('hp_shaft.HPX', 200, units='hp')
        
        self.pyc_add_cycle_param('hpc.bleed_hpc1:frac_P', 0.5)
        self.pyc_add_cycle_param('hpc.bleed_hpc1:frac_work', 0.2)
        self.pyc_add_cycle_param('hpc.bleed_hpc2:frac_P', 0.55)
        self.pyc_add_cycle_param('hpc.bleed_hpc2:frac_work', 0.4)
        self.pyc_add_cycle_param('hpt.bleed_srce1:frac_P', 1.0)
        self.pyc_add_cycle_param('hpt.bleed_srce2:frac_P', 0.0)
        self.pyc_add_cycle_param('hpt.bleed_42:frac_P', 1.0)
        self.pyc_add_cycle_param('lpt.bleed_lpt1:frac_P', 1.0)
        self.pyc_add_cycle_param('lpt.bleed_lpt2:frac_P', 0.0)
        self.pyc_add_cycle_param('lpt.bleed_49:frac_P', 1.0)
        
        self.od_pts = ['OD1', 'OD2', 'OD3', 'OD4', 'OD5', 'OD6', 'OD7', 'OD8', 'OD9', 'OD10', 'OD11', 'OD12', 'OD13', 'OD14', 'OD15', 'OD16', 'OD17', 'OD18', 'OD19']
        self.od_MNs = [0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
        self.od_alts = [0.1, 0.1, 0.1, 1000, 5000, 10000, 18000, 25000, 31000, 36000, 40000, 43000, 46000, 49000, 51000, 53000, 55000, 56500, 58000]
        #self.od_Fns = [38000, 28000, 23000, 21000], units='lbf'), #8950.0    
        self.od_dTs = [18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0]   
        self.od_AB_FAR = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
        self.od_cust_fracW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        for i,pt in enumerate(self.od_pts):
            
            ODpt = self.pyc_add_pnt(pt, sVCE5(design=False, cooling=False))
            
            ODcooling = ODpt.options['cooling']
            
            self.set_input_defaults(pt+'.FAR_bal.T4max', 3500, units='degR')
            self.set_input_defaults(pt+'.FAR_bal.NcMapTgt', 1.0, units='degR')
            
            self.set_input_defaults(pt+'.fc.alt', self.od_alts[i], units='ft')
            self.set_input_defaults(pt+'.fc.MN', self.od_MNs[i], units=None)
            self.set_input_defaults(pt+'.fc.dTs', self.od_dTs[i], units='degR')
            self.set_input_defaults(pt+'.bleeds.cust:frac_W', self.od_cust_fracW[i], units=None)
            #self.set_input_defaults(pt+'.balance.rhs:FAR', 3660, units='degR')
            self.set_input_defaults(pt+'.afterburner.Fl_I:FAR', self.od_AB_FAR[i], units=None)
            
            if not ODcooling:
                if designcooling:            
                    self.connect('DESIGN.balance.lpt_nochrg_cool_frac', pt+'.hpc.bleed_hpc1:frac_W')
                    self.connect('DESIGN.balance.lpt_chrg_cool_frac', pt+'.hpc.bleed_hpc2:frac_W')
                    self.connect('DESIGN.balance.hpt_nochrg_cool_frac', pt+'.bleeds.bleed_srce1:frac_W')
                    self.connect('DESIGN.balance.hpt_chrg_cool_frac', pt+'.bleeds.bleed_srce2:frac_W')
            
                if not designcooling:
                    self.set_input_defaults(pt+'.hpc.bleed_hpc1:frac_W', 0.0001)
                    self.set_input_defaults(pt+'.hpc.bleed_hpc2:frac_W', 0.0001)
                    self.set_input_defaults(pt+'.bleeds.bleed_srce1:frac_W', 0.01)
                    self.set_input_defaults(pt+'.bleeds.bleed_srce2:frac_W', 0.01)
                    
        
        # map scalars
        self.pyc_connect_des_od('fan.s_PR', 'fan.s_PR')
        self.pyc_connect_des_od('fan.s_Wc', 'fan.s_Wc')
        self.pyc_connect_des_od('fan.s_eff', 'fan.s_eff')
        self.pyc_connect_des_od('fan.s_Nc', 'fan.s_Nc')
        self.pyc_connect_des_od('hpc.s_PR', 'hpc.s_PR')
        self.pyc_connect_des_od('hpc.s_Wc', 'hpc.s_Wc')
        self.pyc_connect_des_od('hpc.s_eff', 'hpc.s_eff')
        self.pyc_connect_des_od('hpc.s_Nc', 'hpc.s_Nc')
        self.pyc_connect_des_od('hpt.s_PR', 'hpt.s_PR')
        self.pyc_connect_des_od('hpt.s_Wp', 'hpt.s_Wp')
        self.pyc_connect_des_od('hpt.s_eff', 'hpt.s_eff')
        self.pyc_connect_des_od('hpt.s_Np', 'hpt.s_Np')
        self.pyc_connect_des_od('lpt.s_PR', 'lpt.s_PR')
        self.pyc_connect_des_od('lpt.s_Wp', 'lpt.s_Wp')
        self.pyc_connect_des_od('lpt.s_eff', 'lpt.s_eff')
        self.pyc_connect_des_od('lpt.s_Np', 'lpt.s_Np')
        
        #flow areas
        self.pyc_connect_des_od('inlet.Fl_O:stat:area', 'inlet.area')
        self.pyc_connect_des_od('duct1.Fl_O:stat:area', 'duct1.area')
        self.pyc_connect_des_od('fan.Fl_O:stat:area', 'fan.area')
        self.pyc_connect_des_od('duct21.Fl_O:stat:area', 'duct21.area')
        self.pyc_connect_des_od('splitter.Fl_O1:stat:area', 'splitter.area1')
        self.pyc_connect_des_od('splitter.Fl_O2:stat:area', 'splitter.area2')
        self.pyc_connect_des_od('ductcore.Fl_O:stat:area', 'ductcore.area')
        self.pyc_connect_des_od('ductbypassa.Fl_O:stat:area', 'ductbypassa.area')
        self.pyc_connect_des_od('hpc.Fl_O:stat:area', 'hpc.area')
        self.pyc_connect_des_od('duct3.Fl_O:stat:area', 'duct3.area')
        self.pyc_connect_des_od('bleeds.Fl_O:stat:area', 'bleeds.area')
        self.pyc_connect_des_od('combustor.Fl_O:stat:area', 'combustor.area')
        self.pyc_connect_des_od('duct40.Fl_O:stat:area', 'duct40.area')
        self.pyc_connect_des_od('hpt.Fl_O:stat:area', 'hpt.area')
        self.pyc_connect_des_od('duct42.Fl_O:stat:area', 'duct42.area')
        self.pyc_connect_des_od('lpt.Fl_O:stat:area', 'lpt.area')
        self.pyc_connect_des_od('duct7.Fl_O:stat:area', 'duct7.area')
        self.pyc_connect_des_od('ductbypassb.Fl_O:stat:area', 'ductbypassb.area')
        self.pyc_connect_des_od('mixer.Fl_O:stat:area', 'mixer.area')
        self.pyc_connect_des_od('mixer.Fl_I1_calc:stat:area', 'mixer.Fl_I1_stat_calc.area')
        self.pyc_connect_des_od('duct8.Fl_O:stat:area', 'duct8.area')
        self.pyc_connect_des_od('afterburner.Fl_O:stat:area', 'afterburner.area')


if __name__ == "__main__":

    import time
    from openmdao.api import Problem

    prob = Problem()
    
    prob.model = mp_svce5 = MPsimple_VCE5()

    prob.setup(check=False)

    # initial guesses
    prob['DESIGN.balance.FAR'] = 0.035
    prob['DESIGN.balance.fan_eff'] = 0.89
    prob['DESIGN.balance.hpc_eff'] = 0.87
    prob['DESIGN.balance.lpt_eff'] = 0.90
    prob['DESIGN.balance.hpt_eff'] = 0.90
    prob['DESIGN.balance.BPR'] = 4
    prob['DESIGN.balance.hpt_PR'] = 3.0
    prob['DESIGN.balance.lpt_PR'] = 4.75
    prob['DESIGN.fc.balance.Pt'] = 5.2
    prob['DESIGN.fc.balance.Tt'] = 440.0
    prob['DESIGN.mixer.balance.P_tot']= 40
    
    W_guesses = [550, 550, 560, 480, 430, 330, 270, 230, 200, 180, 210, 185, 170, 160, 140, 125, 120, 115, 110]
    Mixer_P_guesses = [40, 40, 40, 37, 34, 28, 24, 21, 19, 16, 15, 13, 12, 12, 12, 12, 12, 12, 12]  
    
    for i, pt in enumerate(mp_svce5.od_pts):
        # ADP and TOC guesses
        #prob[pt+'.FAR_bal.FAR'] = .034
        prob[pt+'.balance.W'] = W_guesses[i]
        prob[pt+'.balance.BPR'] = 4.3
        prob[pt+'.balance.lp_Nmech'] = 3000
        prob[pt+'.balance.hp_Nmech'] = 6000
        prob[pt+'.mixer.balance.P_tot'] = Mixer_P_guesses[i]
        prob[pt+'.fc.balance.Pt'] = 14
        prob[pt+'.fc.balance.Tt'] = 520
        prob[pt+'.hpt.PR'] = 3
        prob[pt+'.lpt.PR'] = 4.75

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    
    prob.run_model()
    
    for pt in ['DESIGN']+mp_svce5.od_pts:
        viewer(prob, pt)  

    print()
    print("time", time.time() - st)