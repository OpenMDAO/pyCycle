import sys

import numpy as np

import openmdao.api as om

import pycycle.api as pyc


class HBTF(pyc.Cycle):

    def initialize(self):
        # Initialize the model here by setting option variables such as a switch for design vs off-des cases
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')

    def setup(self):
        #Setup the problem by including all the relavant components here - comp, burner, turbine etc
        
        #Create any relavent short hands here:
        thermo_spec = pyc.species_data.janaf #Thermodynamic data specification 
        design = self.options['design']
        
        #Add subsystems to build the engine deck:
        self.pyc_add_element('fc', pyc.FlightConditions(thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        
        self.pyc_add_element('inlet', pyc.Inlet(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        
        # Note variable promotion for the fan -- 
        # the LP spool speed and the fan speed are INPUTS that are promoted:
        # Note here that promotion aliases are used. Here Nmech is being aliased to LP_Nmech
        # in fact for a multi-spool engine you HAVE(?) to alias if you want to promote_inputs
        # check out: http://openmdao.org/twodocs/versions/latest/features/core_features/grouping_components/add_subsystem.html?highlight=alias
        self.pyc_add_element('fan', pyc.Compressor(map_data=pyc.FanMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX,
                                        bleed_names=[], map_extrap=True), promotes_inputs=[('Nmech','LP_Nmech')])
        self.pyc_add_element('splitter', pyc.Splitter(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.pyc_add_element('duct4', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.pyc_add_element('lpc', pyc.Compressor(map_data=pyc.LPCMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX,
                                        map_extrap=True),promotes_inputs=[('Nmech','LP_Nmech')])
        self.pyc_add_element('duct6', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.pyc_add_element('hpc', pyc.Compressor(map_data=pyc.HPCMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX,
                                        bleed_names=['cool1','cool2','cust'], map_extrap=True),promotes_inputs=[('Nmech','HP_Nmech')])
        self.pyc_add_element('bld3', pyc.BleedOut(design=design, bleed_names=['cool3','cool4']))
        self.pyc_add_element('burner', pyc.Combustor(design=design,thermo_data=thermo_spec,
                                        inflow_elements=pyc.AIR_MIX,
                                        air_fuel_elements=pyc.AIR_FUEL_MIX,
                                        fuel_type='Jet-A(g)'))
        self.pyc_add_element('hpt', pyc.Turbine(map_data=pyc.HPTMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX,
                                        bleed_names=['cool3','cool4'], map_extrap=True),promotes_inputs=[('Nmech','HP_Nmech')])
        self.pyc_add_element('duct11', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.pyc_add_element('lpt', pyc.Turbine(map_data=pyc.LPTMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX,
                                        bleed_names=['cool1','cool2'], map_extrap=True),promotes_inputs=[('Nmech','LP_Nmech')])
        self.pyc_add_element('duct13', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.pyc_add_element('core_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv', thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))

        self.pyc_add_element('byp_bld', pyc.BleedOut(design=design, bleed_names=['bypBld']))
        self.pyc_add_element('duct15', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.pyc_add_element('byp_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv', thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        
        #Create shaft instances. Note that LP shaft has 3 ports! => no gearbox
        self.pyc_add_element('lp_shaft', pyc.Shaft(num_ports=3),promotes_inputs=[('Nmech','LP_Nmech')])
        self.pyc_add_element('hp_shaft', pyc.Shaft(num_ports=2),promotes_inputs=[('Nmech','HP_Nmech')])
        self.pyc_add_element('perf', pyc.Performance(num_nozzles=2, num_burners=1))
    
        # Now use the explicit connect method to make connections -- connect(<from>, <to>)
        
        #Connect the inputs to perf group
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('hpc.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('core_nozz.Fg', 'perf.Fg_0')
        self.connect('byp_nozz.Fg', 'perf.Fg_1')
        
        #LP-shaft connections
        self.connect('fan.trq', 'lp_shaft.trq_0')
        self.connect('lpc.trq', 'lp_shaft.trq_1')
        self.connect('lpt.trq', 'lp_shaft.trq_2')
        #HP-shaft connections
        self.connect('hpc.trq', 'hp_shaft.trq_0')
        self.connect('hpt.trq', 'hp_shaft.trq_1')
        #Ideally expanding flow by conneting flight condition static pressure to nozzle exhaust pressure
        self.connect('fc.Fl_O:stat:P', 'core_nozz.Ps_exhaust')
        self.connect('fc.Fl_O:stat:P', 'byp_nozz.Ps_exhaust')
        
        #Create a balance component
        # Balances can be a bit confusing, here's some explanation -
        #   State Variables:
        #           (W)        Inlet mass flow rate to implictly balance thrust
        #                      LHS: perf.Fn  == RHS: Thrust requirement (set when TF is instantiated)
        #
        #           (FAR)      Fuel-air ratio to balance Tt4
        #                      LHS: burner.Fl_O:tot:T  == RHS: Tt4 target (set when TF is instantiated)
        #
        #           (lpt_PR)   LPT press ratio to balance shaft power on the low spool
        #           (hpt_PR)   HPT press ratio to balance shaft power on the high spool
        # Ref: look at the XDSM diagrams in the pyCycle paper and this:
        # http://openmdao.org/twodocs/versions/latest/features/building_blocks/components/balance_comp.html

        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:
            balance.add_balance('W', units='lbm/s', eq_units='lbf')
            #Here balance.W is implicit state variable that is the OUTPUT of balance object
            self.connect('balance.W', 'inlet.Fl_I:stat:W') #Connect the output of balance to the relevant input
            self.connect('perf.Fn', 'balance.lhs:W')       #This statement makes perf.Fn the LHS of the balance eqn.

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')
            
            # Note that for the following two balances the mult val is set to -1 so that the NET torque is zero
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

        else:
            
            #In OFF-DESIGN mode we need to redefine the balances:
            #   State Variables:
            #           (W)        Inlet mass flow rate to balance core flow area
            #                      LHS: core_nozz.Throat:stat:area == Area from DESIGN calculation 
            #
            #           (FAR)      Fuel-air ratio to balance Thrust req.
            #                      LHS: perf.Fn  == RHS: Thrust requirement (set when TF is instantiated)
            #
            #           (BPR)      Bypass ratio to balance byp. noz. area
            #                      LHS: byp_nozz.Throat:stat:area == Area from DESIGN calculation
            #
            #           (lp_Nmech)   LP spool speed to balance shaft power on the low spool
            #           (hp_Nmech)   HP spool speed to balance shaft power on the high spool
            balance.add_balance('FAR', val=0.017, lower=1e-4, eq_units='lbf')
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('perf.Fn', 'balance.lhs:FAR')

            balance.add_balance('W', units='lbm/s', lower=10., upper=1000., eq_units='inch**2')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('core_nozz.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('BPR', lower=2., upper=10., eq_units='inch**2')
            self.connect('balance.BPR', 'splitter.BPR')
            self.connect('byp_nozz.Throat:stat:area', 'balance.lhs:BPR')

            # Again for the following two balances the mult val is set to -1 so that the NET torque is zero
            balance.add_balance('lp_Nmech', val=1.5, units='rpm', lower=500., eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.lp_Nmech', 'LP_Nmech')
            self.connect('lp_shaft.pwr_in_real', 'balance.lhs:lp_Nmech')
            self.connect('lp_shaft.pwr_out_real', 'balance.rhs:lp_Nmech')

            balance.add_balance('hp_Nmech', val=1.5, units='rpm', lower=500., eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.hp_Nmech', 'HP_Nmech')
            self.connect('hp_shaft.pwr_in_real', 'balance.lhs:hp_Nmech')
            self.connect('hp_shaft.pwr_out_real', 'balance.rhs:hp_Nmech')
            
            # Specify the order in which the subsystems are executed:
            
            self.set_order(['balance', 'fc', 'inlet', 'fan', 'splitter', 'duct4', 'lpc', 'duct6', 'hpc', 'bld3', 'burner', 'hpt', 'duct11',
                            'lpt', 'duct13', 'core_nozz', 'byp_bld', 'duct15', 'byp_nozz', 'lp_shaft', 'hp_shaft', 'perf'])
        
        # Set up all the flow connections:
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O', 'fan.Fl_I')
        self.pyc_connect_flow('fan.Fl_O', 'splitter.Fl_I')
        self.pyc_connect_flow('splitter.Fl_O1', 'duct4.Fl_I')
        self.pyc_connect_flow('duct4.Fl_O', 'lpc.Fl_I')
        self.pyc_connect_flow('lpc.Fl_O', 'duct6.Fl_I')
        self.pyc_connect_flow('duct6.Fl_O', 'hpc.Fl_I')
        self.pyc_connect_flow('hpc.Fl_O', 'bld3.Fl_I')
        self.pyc_connect_flow('bld3.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'hpt.Fl_I')
        self.pyc_connect_flow('hpt.Fl_O', 'duct11.Fl_I')
        self.pyc_connect_flow('duct11.Fl_O', 'lpt.Fl_I')
        self.pyc_connect_flow('lpt.Fl_O', 'duct13.Fl_I')
        self.pyc_connect_flow('duct13.Fl_O','core_nozz.Fl_I')
        self.pyc_connect_flow('splitter.Fl_O2', 'byp_bld.Fl_I')
        self.pyc_connect_flow('byp_bld.Fl_O', 'duct15.Fl_I')
        self.pyc_connect_flow('duct15.Fl_O', 'byp_nozz.Fl_I')

        #Bleed flows:
        self.pyc_connect_flow('hpc.cool1', 'lpt.cool1', connect_stat=False)
        self.pyc_connect_flow('hpc.cool2', 'lpt.cool2', connect_stat=False)
        self.pyc_connect_flow('bld3.cool3', 'hpt.cool3', connect_stat=False)
        self.pyc_connect_flow('bld3.cool4', 'hpt.cool4', connect_stat=False)
        
        #Specify solver settings:
        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-8
        newton.options['rtol'] = 1e-8
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 50
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False
        # ls = newton.linesearch = BoundsEnforceLS()
        ls = newton.linesearch = om.ArmijoGoldsteinLS()
        ls.options['maxiter'] = 3
        ls.options['bound_enforcement'] = 'scalar'
        # ls.options['print_bound_enforce'] = True

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
        FAR = prob[pt+'.balance.FAR']

    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC      BPR ", file=file, flush=True)
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f  %7.3f" %(MN, prob[pt+'.fc.alt'],prob[pt+'.inlet.Fl_O:stat:W'],prob[pt+'.perf.Fn'],prob[pt+'.perf.Fg'],prob[pt+'.inlet.F_ram'],prob[pt+'.perf.OPR'],prob[pt+'.perf.TSFC'], prob[pt+'.splitter.BPR']), file=file, flush=True)


    fs_names = ['fc.Fl_O', 'inlet.Fl_O', 'fan.Fl_O', 'splitter.Fl_O1', 'splitter.Fl_O2',
                'duct4.Fl_O', 'lpc.Fl_O', 'duct6.Fl_O', 'hpc.Fl_O', 'bld3.Fl_O', 'burner.Fl_O',
                'hpt.Fl_O', 'duct11.Fl_O', 'lpt.Fl_O', 'duct13.Fl_O', 'core_nozz.Fl_O', 'byp_bld.Fl_O',
                'duct15.Fl_O', 'byp_nozz.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    comp_names = ['fan', 'lpc', 'hpc']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.print_compressor(prob, comp_full_names, file=file)

    pyc.print_burner(prob, [f'{pt}.burner'])

    turb_names = ['hpt', 'lpt']
    turb_full_names = [f'{pt}.{t}' for t in turb_names]
    pyc.print_turbine(prob, turb_full_names, file=file)

    noz_names = ['core_nozz', 'byp_nozz']
    noz_full_names = [f'{pt}.{n}' for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)

    shaft_names = ['hp_shaft', 'lp_shaft']
    shaft_full_names = [f'{pt}.{s}' for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=file)

    bleed_names = ['hpc', 'bld3', 'byp_bld']
    bleed_full_names = [f'{pt}.{b}' for b in bleed_names]
    pyc.print_bleed(prob, bleed_full_names, file=file)


class MPhbtf(pyc.MPCycle):

    def setup(self):

        self.pyc_add_pnt('DESIGN', HBTF()) # Create an instace of the High Bypass ratio Turbofan
        self.pyc_add_cycle_param('inlet.ram_recovery', 0.9990)
        self.pyc_add_cycle_param('duct4.dPqP', 0.0048)
        self.pyc_add_cycle_param('duct6.dPqP', 0.0101)
        self.pyc_add_cycle_param('burner.dPqP', 0.0540)
        self.pyc_add_cycle_param('duct11.dPqP', 0.0051)
        self.pyc_add_cycle_param('duct13.dPqP', 0.0107)
        self.pyc_add_cycle_param('duct15.dPqP', 0.0149)
        self.pyc_add_cycle_param('core_nozz.Cv', 0.9933)
        self.pyc_add_cycle_param('byp_bld.bypBld:frac_W', 0.005)
        self.pyc_add_cycle_param('byp_nozz.Cv', 0.9939)
        self.pyc_add_cycle_param('hpc.cool1:frac_W', 0.050708)
        self.pyc_add_cycle_param('hpc.cool1:frac_P', 0.5)
        self.pyc_add_cycle_param('hpc.cool1:frac_work', 0.5)
        self.pyc_add_cycle_param('hpc.cool2:frac_W', 0.020274)
        self.pyc_add_cycle_param('hpc.cool2:frac_P', 0.55)
        self.pyc_add_cycle_param('hpc.cool2:frac_work', 0.5)
        self.pyc_add_cycle_param('bld3.cool3:frac_W', 0.067214)
        self.pyc_add_cycle_param('bld3.cool4:frac_W', 0.101256)
        self.pyc_add_cycle_param('hpc.cust:frac_P', 0.5)
        self.pyc_add_cycle_param('hpc.cust:frac_work', 0.5)
        self.pyc_add_cycle_param('hpt.cool3:frac_P', 1.0)
        self.pyc_add_cycle_param('hpt.cool4:frac_P', 0.0)
        self.pyc_add_cycle_param('lpt.cool1:frac_P', 1.0)
        self.pyc_add_cycle_param('lpt.cool2:frac_P', 0.0)
        self.pyc_add_cycle_param('hp_shaft.HPX', 250.0, units='hp')

        pts = ['OD'] #,'OD2','OD3','OD4']

        for i_OD, pt in enumerate(pts):
            ODpt = self.pyc_add_pnt(pt, HBTF(design=False))

        #Connect all DESIGN map scalars to the off design cases
        self.pyc_connect_des_od('fan.s_PR', 'fan.s_PR')
        self.pyc_connect_des_od('fan.s_Wc', 'fan.s_Wc')
        self.pyc_connect_des_od('fan.s_eff', 'fan.s_eff')
        self.pyc_connect_des_od('fan.s_Nc', 'fan.s_Nc')
        self.pyc_connect_des_od('lpc.s_PR', 'lpc.s_PR')
        self.pyc_connect_des_od('lpc.s_Wc', 'lpc.s_Wc')
        self.pyc_connect_des_od('lpc.s_eff', 'lpc.s_eff')
        self.pyc_connect_des_od('lpc.s_Nc', 'lpc.s_Nc')
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
        
        #Set up the RHS of the balances!
        self.pyc_connect_des_od('core_nozz.Throat:stat:area','balance.rhs:W')
        self.pyc_connect_des_od('byp_nozz.Throat:stat:area','balance.rhs:BPR')

        self.pyc_connect_des_od('inlet.Fl_O:stat:area', 'inlet.area')
        self.pyc_connect_des_od('fan.Fl_O:stat:area', 'fan.area')
        self.pyc_connect_des_od('splitter.Fl_O1:stat:area', 'splitter.area1')
        self.pyc_connect_des_od('splitter.Fl_O2:stat:area', 'splitter.area2')
        self.pyc_connect_des_od('duct4.Fl_O:stat:area', 'duct4.area')
        self.pyc_connect_des_od('lpc.Fl_O:stat:area', 'lpc.area')
        self.pyc_connect_des_od('duct6.Fl_O:stat:area', 'duct6.area')
        self.pyc_connect_des_od('hpc.Fl_O:stat:area', 'hpc.area')
        self.pyc_connect_des_od('bld3.Fl_O:stat:area', 'bld3.area')
        self.pyc_connect_des_od('burner.Fl_O:stat:area', 'burner.area')
        self.pyc_connect_des_od('hpt.Fl_O:stat:area', 'hpt.area')
        self.pyc_connect_des_od('duct11.Fl_O:stat:area', 'duct11.area')
        self.pyc_connect_des_od('lpt.Fl_O:stat:area', 'lpt.area')
        self.pyc_connect_des_od('duct13.Fl_O:stat:area', 'duct13.area')
        self.pyc_connect_des_od('byp_bld.Fl_O:stat:area', 'byp_bld.area')
        self.pyc_connect_des_od('duct15.Fl_O:stat:area', 'duct15.area')

                # FOR DESIGN
        # Note that here the values we are setting are actually DESIGN INPUTS/ FLIGHT CONDITIONS

        # ====== START DECLARING DESIGN VARIABLES ====== 

        # Component level setup
        # --- INLET -----
        self.set_input_defaults('DESIGN.inlet.MN', 0.751)

        # ---------------
        # ----- FAN -----
        self.set_input_defaults('DESIGN.fan.MN', 0.4578)

        # ---------------
        # --- SPLITTER ---
        self.set_input_defaults('DESIGN.splitter.BPR', 5.105)
        self.set_input_defaults('DESIGN.splitter.MN1', 0.3104)
        self.set_input_defaults('DESIGN.splitter.MN2', 0.4518)

        # ---------------
        # --- DUCT 4 -----
        self.set_input_defaults('DESIGN.duct4.MN', 0.3121)

        # ---------------
        # --- LPC -----
        # self.set_input_defaults('DESIGN.lpc.eff', 0.9243)
        self.set_input_defaults('DESIGN.lpc.MN', 0.3059)

        # ---------------
        # --- DUCT 6 -----
        self.set_input_defaults('DESIGN.duct6.MN', 0.3563),

        # ---------------
        # ---  HPC -----
        self.set_input_defaults('DESIGN.hpc.MN', 0.2442),

        # ---------------
        # --- BLEED -----
        self.set_input_defaults('DESIGN.bld3.MN', 0.3000)

        # ---------------
        # --- BURNER -----
        self.set_input_defaults('DESIGN.burner.MN', 0.1025),

        # ---------------
        # --- HPT -----
        self.set_input_defaults('DESIGN.hpt.MN', 0.3650),

        # ---------------
        # --- DUCT -----
        self.set_input_defaults('DESIGN.duct11.MN', 0.3063),

        # ---------------
        # --- LPT -----
        self.set_input_defaults('DESIGN.lpt.MN', 0.4127),

        # ---------------
        # --- DUCT 13 -----
        self.set_input_defaults('DESIGN.duct13.MN', 0.4463),

        # ---------------
        # --- BLEED -----
        self.set_input_defaults('DESIGN.byp_bld.MN', 0.4489),

        # ---------------
        # --- DUCT 15 -----
        self.set_input_defaults('DESIGN.duct15.MN', 0.4589),

        # ---------------
        # --- LP SHAFT -----
        self.set_input_defaults('DESIGN.LP_Nmech', 4666.1, units='rpm'),

        # ---------------
        # --- HP SHAFT -----
        self.set_input_defaults('DESIGN.HP_Nmech', 14705.7, units='rpm'),

        # --- Set up bleed values -----
        self.set_input_defaults('DESIGN.hpc.cust:frac_W', 0.0445),





if __name__ == "__main__":

    import time

    prob = om.Problem()

    prob.model = MPhbtf()

    prob.setup(check=False)

    ####Values that won't allow set_input_defaults to be called:
    prob.set_val('DESIGN.fan.PR', 1.685)
    prob.set_val('DESIGN.fan.eff', 0.8948)

    prob.set_val('DESIGN.lpc.PR', 1.935)

    prob.set_val('DESIGN.lpc.eff', 0.9243)

    prob.set_val('DESIGN.hpc.PR', 9.369),
    prob.set_val('DESIGN.hpc.eff', 0.8707),

    prob.set_val('DESIGN.hpt.eff', 0.8888),

    prob.set_val('DESIGN.lpt.eff', 0.8996),

    ####Values that are unique to each run

    #Flight conditions
    prob.set_val('DESIGN.fc.alt', 35000., units='ft')
    prob.set_val('DESIGN.fc.MN', 0.8)

    #Target Tt4 and Fn_design for the balances
    prob.set_val('DESIGN.balance.rhs:FAR', 2857, units='degR')
    prob.set_val('DESIGN.balance.rhs:W', 5500.0, units='lbf') 

    # OFF DESIGN
    # The arrays represent multiple flight conditions.
    pts = ['OD'] #,'OD2','OD3','OD4'] 
    OD_MN = [0.8, 0.8, 0.25, 0.00001]
    OD_alt = [35000.0, 35000.0, 0.0, 0.0]
    OD_FAR = [5500.0, 5970.0, 22590.0, 27113.0]
    OD_dTs = [0.0, 0.0, 27.0, 27.0]
    OD_W = [0.0445, 0.0422, 0.0177, 0.0185]

    for i_OD, pt in enumerate(pts):
        prob.set_val(pt+'.fc.MN', OD_MN[i_OD]),
        prob.set_val(pt+'.fc.alt', OD_alt[i_OD], units='ft'),
        prob.set_val(pt+'.balance.rhs:FAR', OD_FAR[i_OD], units='lbf'), #8950.0
        prob.set_val(pt+'.fc.dTs', OD_dTs[i_OD], units='degR')
        prob.set_val(pt+'.hpc.cust:frac_W', OD_W[i_OD])

    ####Value that are initial guesses
    prob['DESIGN.balance.FAR'] = 0.025
    prob['DESIGN.balance.W'] = 100.
    prob['DESIGN.balance.lpt_PR'] = 4.0
    prob['DESIGN.balance.hpt_PR'] = 3.0
    prob['DESIGN.fc.balance.Pt'] = 5.2
    prob['DESIGN.fc.balance.Tt'] = 440.0

    W_guesses = [300, 300, 700, 700]
    for i, pt in enumerate(pts):
        # ADP and TOC guesses
        prob[pt+'.balance.FAR'] = 0.02467
        prob[pt+'.balance.W'] = W_guesses[i]
        prob[pt+'.balance.BPR'] = 5.105
        prob[pt+'.balance.lp_Nmech'] = 5000 # 4666.1
        prob[pt+'.balance.hp_Nmech'] = 15000 # 14705.7
        # prob[pt+'.fc.balance.Pt'] = 5.2
        # prob[pt+'.fc.balance.Tt'] = 440.0
        prob[pt+'.hpt.PR'] = 3.
        prob[pt+'.lpt.PR'] = 4.
        prob[pt+'.fan.map.RlineMap'] = 2.0
        prob[pt+'.lpc.map.RlineMap'] = 2.0
        prob[pt+'.hpc.map.RlineMap'] = 2.0

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()
    prob.model.DESIGN.list_outputs(residuals=True, residuals_tol=1e-2)

    for pt in ['DESIGN']+pts:
        viewer(prob, pt)

    print()
    print("Run time", time.time() - st)