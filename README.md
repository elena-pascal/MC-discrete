# MC-discrete
Discrete inelastic electron scattering Monte Carlo implementation

MC/`doScatter.py` calls the rest of the code according to the parameters set in inputFile/`input.file`:
* number of electron per process : `num_el`
* incident beam energy: `E0`
* electrons are tracked from energy E_0 to a set minimum energy: `Emin`
* material with which the electrons interact `material`; so far only free electron systems are implemented: Al, Si, Cu, Au
* the tilt of the sample in degrees defined in the same way as in EMsoft: `s_tilt`
* scattering mode, `mode`, can be continuous or discrete (DS)



tests/`testThings.py` runs some statistical tests for the stochastic patameters, set up follows tests/`inputTest.file`

`doScatter.py` calls `multiScattering.py` which spawn a multiprocessing job for one electron trajectory.
