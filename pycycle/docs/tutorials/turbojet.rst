--------
TurboJet
--------

The performance of all jet engines can be modeled by setting up an appropriate model of the thermodynamic cycle that a particular engine is based on. 
One of the simplest kinds of jet engines is the single spool turbojet engine. 
The first ever production turbojet engine, the Junkers Jumo 004, was a single spool turbojet that was used to power the ME-262 jet fighter at the end of world war 2. 
By "single spool", we mean that there is a one spinning shaft that connects a single compressor to a single turbine. 

The relative simplicity that is an inherent feature of the single spool turbojet makes it a good place to start learning how to model the performance of jet engines. 

[Image of model block diagram]

on-design vs. off-design
------------------------
We're going to walk you through a run script that builds a single spool turbojet model and runs it in both **on-design** and **off-design** modes. 
All pyCycle models will be run in both modes, and it is important to understand what each one is for and how the relate to each other. 

explain on-design and off-design


Preamble 
--------
Import statements

Creating the turbojet model
---------------------------
 
    Adding cycle elements
    
    Setting thermodynamic properties

    Connecting flow stations

    Connecting cycle elements
    
    Setting up balances

    Setting execution order

    Add/setup Newton solver


Configuring output viewer
--------------------------

Executing the model
-------------------

Example results (page viewer)
-----------------------------