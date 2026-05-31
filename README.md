# Part III Project - Time-Dependent CP Violation at the Future Circular Collider

This project studies the time-dependent CP violation in the decay channel $B^{0} \rightarrow J/\psi K^{0}_{S}$, from which the CKM angle $\beta$ can be determined.
In this decay channel, the CP violation parameters are $C_{f} \approx 0$ and $S_{f} = \sin(2\beta)$ to leading order. The CP asymmetry is simply $A_{CP} \approx S_{f} sin(x \Gamma t)$.
The goal of this project is to provide an estimate of the statistical precision in $\beta$ measurements at the proposed FCC-ee using toy Monte Carlo simulations.
This model is based on a simultaneous decay-time and incariant-mass fit of flavour-tagged events. Detector and selection effects such as signal acceptance, background contributions, and flavour mistagging have been incorporated.
This is the code repository for the project, including the toy Monte Carlo simulation framework and other related analyses.


## Toy Monte Carlo Simulation

In the "simulation" directory, "simulation.py" performs the simulation and writes the results, and then running "analysis.py" will produce relevant plots.

The file "single.py" performs a single pseudo-experiment and generates truth-level PDF plots and the iminuit output for the run.


## Fisher Information Approach

An alternative approach to the project is predicting the precision anlytically via the Fisher information, which is beyond the scope of this project. This part is included here only for demonstration purposes.

The files fisher_information_integration.py gives the exact per-event Fisher information of the CP violation parameter $S_{f}$ in the statistical model. The file fisher_information_analysis.py then computes the predicted precision in $S_{f}$ under significant simplification assumptions.
