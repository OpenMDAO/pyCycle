[![Build Status](https://travis-ci.org/OpenMDAO/pyCycle.svg?branch=master)](https://travis-ci.org/OpenMDAO/pyCycle)

# pyCycle
--------------

This is a thermodynamic cycle modeling library, designed primarily to model jet engine performance. 
It is built on top of the OpenMDAO framework and the design is heavily inspired by NASA's NPSS software.
You need to be at least familiar with either OpenMDAO or NPSS in order to be successful in using this library. 

Disclosure: The docs are nearly non-existent. We're hoping to improve this, but for the moment this is what you get. 
We suggest you look in the examples folder for some indications of how to run this code. 
Also, you can should [the paper on pyCycle](https://www.mdpi.com/2226-4310/6/8/87/pdf) which goes into a lot of detail that is very relevant. 


# OpenMDAO Version Compatibility
----------------------------------
pyCycle is built on top of OpenMDAO, and thus depends on it. Here is the OpenMDAO version you need for the specific versions of pyCycle


pyCycle 3.0.0 -- 2.80 >= OpenMDAO <=3.1.1
pyCycle 3.1.0 -- 3.20 >= OpenMDAO


## Citation

If you use pyCycle, please cite this paper: 

*E. S. Hendricks and J. S. Gray, “Pycycle: a tool for efficient optimization of gas turbine engine cycles,” Aerospace, vol. 6, iss. 87, 2019.*

    @article{Hendricks2019, 
    author="Eric S. Hendricks and Justin S. Gray" , 
    title = "pyCycle: A Tool for Efficient Optimization of Gas Turbine Engine Cycles", 
    journal = "Aerospace", 
    year = "2019", 
    day = "8", 
    month = "August",
    volume = {6},  
    number = {87},
    doi = {10.3390/aerospace6080087},
    }

## Installation 

clone this repo, then use pip to install: 

    pip install pyCycle