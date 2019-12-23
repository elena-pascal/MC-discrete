# MC-discrete
Discrete inelastic electron scattering Monte Carlo implementation

`doScatter.py` calls the rest of the code according to the info in `inputFile.py`

`testThings.py` runs some statistical tests for the stochastic patameters, set up follows `inputTest.py`

`doScatter.py` calls `multiScattering.py` which spawn a multiprocessing job for one electron trajectory.
