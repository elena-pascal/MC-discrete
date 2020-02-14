# MC-discrete (very alpha)

Discrete inelastic electron scattering Monte Carlo python implementation

## Required libraries
* `numpy`
* `scipy`
* `scimath`
* `math`
* `multiprocessing`
* `dask`


MC/`doScatter.py` calls the rest of the code according to the parameters set in inputFile/`input.file`. The parametes in the input file are:
* number of electron per process : `num_el`
* incident beam energy: `E0`
* electrons are tracked from energy E_0 to a set minimum energy: `Emin`
* material with which the electrons interact `material`; so far only free electron systems are implemented: Al, Si, Cu, Au
* the tilt of the sample in degrees defined in the same way as in EMsoft: `s_tilt`
* scattering mode, `mode`, can be _continuous_ or discrete (_DS_)
* if continuous mode is chosen then three implementatins are available for the `Bethe_model`: _classical_, _JL_ or _explicit_

* two tolerances can be set for the determining the accuracy of the integration tables for the discrete model: `tol_E` for the accuracy of the trapezoidal  integrals and `tol_W` for the steps of energy loss determining the accuracy in that direction
* `Wc` is a free parameter needed for Moller scattering and represents the minimum energy loss allowed for this channel.

* the maximum number of scatterings can also be set thorough `maxScatt` (though this tends to be disables untill I figure out the physics of setting this)

* the output is made of two dataframes in a hdf5 table with columns set here. `electron_output` contains all the electron parameters of interenst and `scatter_output` the scattering values. 

## Set up
Bacause I moved the python files in separate folders and used namespaces to call them, python needs to know the path we're in:

`export PYTHONPATH=[where the repo is]/MC-discrete/`


## Tests

tests/`testThings.py` runs some statistical tests for the stochastic patameters, set up follows tests/`inputTest.file`

Thing to finish:
* The set up should only be called once for all the tests in the class.
* Gryz scatter test
* Quinn scatter test
* the set unit situatuation should be moved to a test

## Structure
`doScatter.py` calls `scatterMultiEl_cont` or `scatterMultiEl_DS` from `multiScattering.py` depending on the mode set. These functions will spawn a multiprocessing job for one electron trajectory for the number of electrons set. For every trajectory a new electron instance is spawned which is allowed to suffer instances of scattering untill the conditions for basckscattering (/transmission in the future), absorbtion (or if set max number of scatterings) are met. 

At every scattering the parameters set in `scatter_output` as saved to a queue and finally added to the output dataset. 
For every electron the parameters set i `electron_output` are similarly saved to a queue and then to a output dataset. 

## Current issues
hdf5 locks for largish data (electrons>500000).  
