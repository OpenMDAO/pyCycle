#3.9.9

** Backwards incompatible API changes **

The thermo refactorization continues. 
A lot more internal modularization happened, 
and though it mostly does not impact user facing APIs for model building, 
it does affect strongly the way that Elements are coded. 
The goal continues to be to change things to make it possible to add swappable thermo but we are not quite there yet. 

* `Element` base class has been introduced and must be used for anything that contains fluid ports. 

* `Element` now have a `pyc_setup_output_ports` method that must be implemented to propagate 
port setup data downstream. 

* `pyc_add_element` has been deprecated. It was just added in V3.4.0, so it had a short but exciting life. 
It is not needed any more because the presence of the `Element` class made it possible to handle all the details with the standard `add_subsystem` call. 
It seems better to stick to stock OpenMDAO APIs where possible, so thats what we're going to do. 
The old method is getting deprecated, and you'll get a noisy warning. Expect it to go away in V4.0

#3.4.0

* new `MPCycle` (stands for MultiPoint Cycle) and `Cycle` classes that you can optionally use to simplify your models and reduce boiler plate code associated with connecting data between design and off-design instances. 
* major refactor of the thermodynamics library that won't directly affect your models, but is a major cleanup of the core thermo implementation. Though it is fully backwards compatible with 3.2.0, you may notice some small numerical differences due to slightly different solver structure for the CEA solver. 

The thermo refactor has been specifically designed to make it easier to swap between multiple thermodynamics analyses 
(i.e. simpler ones than CEA). 
No other thermodynamic solvers are currently implemented, but they will be coming in future versions. 

The features that will allow you to select from multiple thermodynamics libraries will be integrated in pyCycle 4.0.0. 
This version will likely be slightly backwards incompatible, in terms of how you instantiate the elements. 
If possible we'll provide a deprecations, but regardless it should be fairly simple to update to the new APIs. 