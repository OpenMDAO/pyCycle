import numpy as np

from openmdao.api import ExplicitComponent


class Shaft(ExplicitComponent):

    """Calculates power balance for shaft"""

    def initialize(self):
        self.options.declare('num_ports', default=2,
                              desc="number shaft connections to make")

    def setup(self):

        num_ports = self.options['num_ports']

        self.add_input('Nmech', val = 1000.0, units="rpm")
        self.add_input('HPX', val = 0.0, units='hp')
        self.add_input('fracLoss', val = 0.0)

        self.add_output('trq_in', val=1.0, units='ft*lbf')
        self.add_output('trq_out', val=1.0, units='ft*lbf')
        self.add_output('trq_net', val=1.0, units='ft*lbf')
        self.add_output('pwr_in', val=1.0, units='hp')
        self.add_output('pwr_in_real', val=1.0, units='hp')
        self.add_output('pwr_out', val=1.0, units='hp')
        self.add_output('pwr_out_real', val=1.0, units='hp')
        self.add_output('pwr_net', val=1.0, units='hp')

        HP_to_FT_LBF_per_SEC = 550
        self.convert = 2. * np.pi / 60. / HP_to_FT_LBF_per_SEC

        self.trq_vars = []
        for i in range(num_ports):
            trq_var_name = 'trq_{:d}'.format(i)
            self.add_input(trq_var_name, val=0., units='ft*lbf')

            self.trq_vars.append(trq_var_name)

            self.declare_partials(['trq_in', 'trq_out', 'pwr_in', 'pwr_out'], trq_var_name)

        self.declare_partials('trq_net', '*')
        self.declare_partials('pwr_net', '*')
        self.declare_partials(['pwr_in', 'pwr_out', 'pwr_in_real', 'pwr_out_real'], '*')

    def compute(self, inputs, outputs):

        fracLoss = inputs['fracLoss']
        HPX = inputs['HPX']
        Nmech = inputs['Nmech']

        trq_in = 0
        trq_out = 0

        for trq_var in self.trq_vars:
            trq = inputs[trq_var]
            if trq >= 0:
                trq_in += trq
            else:
                trq_out += trq

        trq_net = trq_in * (1. - fracLoss) + trq_out - HPX / (Nmech * self.convert)
        outputs['trq_net'] = trq_net

        outputs['trq_in'] = trq_in
        outputs['trq_out'] = trq_out
        outputs['pwr_in'] = trq_in * Nmech * self.convert
        outputs['pwr_out'] = trq_out * Nmech * self.convert
        outputs['pwr_net'] = trq_net * Nmech * self.convert
        outputs['pwr_in_real'] = trq_in * (1. - fracLoss) * Nmech * self.convert
        outputs['pwr_out_real'] = trq_out * Nmech * self.convert - HPX

    def compute_partials(self, inputs, J):
        num_ports = self.options['num_ports']

        PortTrqs = [inputs['trq_%d'%i] for i in range(num_ports)]

        fracLoss = inputs['fracLoss']
        HPX = inputs['HPX']
        Nmech = inputs['Nmech']

        trq_in = 0
        trq_out = 0

        for trq_var in self.trq_vars:
            trq = inputs[trq_var]
            if trq >= 0:
                trq_in += trq
            else:
                trq_out += trq

        J['trq_net', 'Nmech'] = HPX * Nmech ** (-2.) / self.convert
        J['trq_net', 'HPX'] = -1. / (Nmech * self.convert)
        J['trq_net', 'fracLoss'] =  -trq_in

        J['pwr_in', 'Nmech'] = trq_in * self.convert

        J['pwr_out', 'Nmech'] =  trq_out * self.convert

        J['pwr_in_real', 'Nmech'] = trq_in * self.convert * (1 - fracLoss)
        J['pwr_in_real', 'fracLoss'] = -trq_in * self.convert * Nmech

        J['pwr_out_real', 'Nmech'] =  trq_out * self.convert
        J['pwr_out_real', 'HPX'] =  -1

        J['pwr_net', 'Nmech'] = trq_in * \
            (1 - fracLoss) * self.convert + trq_out * self.convert
        J['pwr_net', 'HPX'] = -1
        J['pwr_net', 'fracLoss'] = -trq_in * Nmech * self.convert

        for i in range(num_ports):
            trq_var_name = 'trq_%d'%i

            if PortTrqs[i] >= 0:
                J['trq_in', trq_var_name]= 1.0
                J['trq_out', trq_var_name]= 0.0
                J['trq_net', trq_var_name]= 1 - fracLoss
                J['pwr_in', trq_var_name]= Nmech * self.convert
                J['pwr_out', trq_var_name] = 0.0
                J['pwr_in_real', trq_var_name]= Nmech * self.convert * (1 - fracLoss)
                J['pwr_out_real', trq_var_name] = 0.0
                J['pwr_net', trq_var_name]= Nmech * \
                    self.convert * (1 - fracLoss)

            elif PortTrqs[i] < 0:
                J['trq_out', trq_var_name] = 1.0
                J['trq_in', trq_var_name] = 0.0
                J['trq_net', trq_var_name] = 1.0
                J['pwr_in', trq_var_name]= 0.
                J['pwr_out', trq_var_name] = Nmech * self.convert
                J['pwr_in_real', trq_var_name]= 0.
                J['pwr_out_real', trq_var_name] = Nmech * self.convert
                J['pwr_net', trq_var_name] = Nmech * self.convert


if __name__ == "__main__":
    from openmdao.api import Problem,  Group

    p = Problem()
    p.model = Group()
    p.model.add_subsystem("shaft", Shaft(10))

    p.setup()
    p.run_model()

    #print(p['shaft.PortTrqs'])
