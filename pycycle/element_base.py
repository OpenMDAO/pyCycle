import numpy as np
import openmdao.api as om

from pycycle.flow_in import FlowIn
from pycycle.thermo.thermo import Thermo, ThermoAdd
from pycycle.constants import ALLOWED_THERMOS


class Element(om.Group): 
    """
    Custom pyCycle group for anything that requires input or output ports
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.Fl_I_data = {}
        self.Fl_O_data = {}

    def initialize(self): 

        self.options.declare('design', default=True, 
                              desc='Switch between on-design and off-design calculation.')
        self.options.declare('thermo_data', default=False,
                              desc='thermodynamic data specific to this element', recordable=False)
        self.options.declare('thermo_method', default='CEA', values=ALLOWED_THERMOS,
                              desc='Method for computing thermodynamic properties')

    def copy_flow(self, src_port, output_port): 
        """
        Copy the flow data from `src_from` port to `target_to` port

        src_port: str or <ThermoAdd>
            the name of the input port to copy from, or the ThermoAdd instance to query
        """

        if isinstance(src_port, str): 
            self.Fl_O_data[output_port] = self.Fl_I_data[src_port]
        elif isinstance(src_port, ThermoAdd): 
            self.Fl_O_data[output_port] = src_port.output_port_data()
        else: 
            raise ValueError('copy_from argument must be either a string that is '
                             'the name of an input port, or a ThermoAdd instance')

    def init_output_flow(self, port_name, port_data): 
        """
        Initialize the given output port with the pord_data
        """

        if isinstance(port_data, ThermoAdd): 
            self.Fl_O_data[port_name] = port_data.output_port_data()
        else: 
            self.Fl_O_data[port_name] = port_data


    # TODO: at end of setup, compare all the ports to whats in the port data and make sure that there is nothing missing






