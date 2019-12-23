# MC-discrete
Discrete inelastic electron scattering Monte Carlo implementation

MC/`doScatter.py` calls the rest of the code according to the parameters set in inputFile/`input.file`

tests/`testThings.py` runs some statistical tests for the stochastic patameters, set up follows tests/`inputTest.file`

`doScatter.py` calls `multiScattering.py` which spawn a multiprocessing job for one electron trajectory.
