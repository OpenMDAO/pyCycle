import openmdao.api as om


class Gearbox(om.ImplicitComponent):
    """Gearbox component based on N+3 model"""

    def initialize(self):
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')

    def setup(self):

        design = self.options['design']

        self.add_input('N_in', val=1000.0, units='rpm', desc='Shaft speed entering gearbox')
        self.add_input('N_out', val=1000.0, units='rpm', desc='Shaft speed exiting gearbox')
        self.add_input('eff', val=1.0, units=None, desc='Gearbox transmission efficiency')

        self.add_output('trq_in', val=1.0, units='ft*lbf', desc='Torque entering gearbox')
        self.add_output('trq_out', val=1.0, units='ft*lbf', desc='Torque exiting gearbox')

        if design:

            self.add_input('trq_base', val=1.0, units='ft*lbf', desc='Base torque value')
            self.add_output('gear_ratio', val=1.0, units=None, desc='Gear ratio (N_out/N_in)')

            self.declare_partials('gear_ratio', ['N_in','N_out'])
            self.declare_partials('gear_ratio', 'gear_ratio', val=-1.0)
            self.declare_partials('trq_in', ['N_in','N_out'])

        else:
            self.add_input('gear_ratio', val=1.0, units=None, desc='Gear ratio (N_out/N_in)')
            self.add_output('trq_base', val=1.0, units='ft*lbf', desc='Base torque value')

            self.declare_partials('trq_base', ['N_in','gear_ratio'])
            self.declare_partials('trq_base', 'N_out', val=1.0)
            self.declare_partials('trq_in','gear_ratio')

        self.declare_partials('trq_in', ['trq_base','eff'])
        self.declare_partials('trq_in', 'trq_in', val=-1.0)
        self.declare_partials('trq_out', 'trq_base', val=1.0)
        self.declare_partials('trq_out', 'trq_out', val=-1.0)


    def solve_nonlinear(self, inputs, outputs):

        design = self.options['design']

        if design:
            N_in, N_out, eff, trq_base = inputs.split_vars()
            gear_ratio = N_out / N_in
            trq_in = -trq_base*eff*N_out / N_in
            trq_out = trq_base
            outputs.join_vals(trq_in, trq_out, gear_ratio)
        else:
            _, _, eff, gear_ratio = inputs.split_vars()
            outputs['trq_in'] = -outputs['trq_base']*eff*gear_ratio
            outputs['trq_out'] = outputs['trq_base']

    def apply_nonlinear(self, inputs, outputs, resids):

        design = self.options['design']

        if design:
            N_in, N_out, eff, trq_base = inputs.split_vars()
            resids['gear_ratio'] = N_out / N_in - outputs['gear_ratio']
            resids['trq_in'] = -trq_base*eff*N_out / N_in - outputs['trq_in']
            resids['trq_out'] = trq_base - outputs['trq_out']

        else:
            N_in, N_out, eff, gear_ratio = inputs.split_vars()
            resids['trq_base'] = N_out - N_in * gear_ratio
            resids['trq_in'] = -outputs['trq_base']*eff*gear_ratio - outputs['trq_in']
            resids['trq_out'] = outputs['trq_base'] - outputs['trq_out']

    def linearize(self, inputs, outputs, J):

        design = self.options['design']

        if design:
            N_in, N_out, eff, trq_base = inputs.split_vars()
            J['gear_ratio','N_in'] = -N_out/N_in**2
            J['gear_ratio','N_out'] = 1.0/N_in

            J['trq_in','trq_base'] = -eff * N_out / N_in
            J['trq_in','eff'] = -trq_base * N_out / N_in
            J['trq_in','N_in'] = trq_base * eff * N_out / N_in**2
            J['trq_in','N_out'] = -trq_base * eff / N_in
        else:
            N_in, N_out, eff, gear_ratio = inputs.split_vars()
            J['trq_base','N_in'] = -gear_ratio
            J['trq_base','gear_ratio'] = -N_in

            J['trq_in','trq_base'] = -eff*gear_ratio
            J['trq_in','eff'] = -outputs['trq_base']*gear_ratio
            J['trq_in','gear_ratio'] = -outputs['trq_base']*eff


if __name__ == "__main__":

    p = om.Problem()

    inputs = p.model.add_subsystem('inputs',om.IndepVarComp(), promotes=['*'])
    inputs.add_output('eff', 1.0)
    inputs.add_output('N_in', 6772.0, units='rpm')
    inputs.add_output('N_out', 2184.5, units='rpm')
    inputs.add_output('trq_base', 23711.1, units='ft*lbf')
    # inputs.add_output('gear_ratio', 0.322578263438, units=None)


    p.model.add_subsystem('gearbox', Gearbox(design=True), promotes=['*'])

    p.setup()
    # p['trq_base'] = 23711.1
    p.run_model()

    p.check_partials(compact_print=True)

    print(p['trq_in'][0])
    print(p['trq_out'][0])
    print(p['gear_ratio'][0])

    print(p['N_in'][0]*p['trq_in'][0]/5252.0)
    print(p['N_out'][0]*p['trq_out'][0]/5252.0)
