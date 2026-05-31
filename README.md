# Part III Project - Time-Dependent CP Violation at the Future Circular Collider

This project studies the time-dependent CP violation in the decay channel $B^{0} \rightarrow J/\psi K^{0}_{S}$, from which the CKM angle $\beta$ can be determined.
In this channel, we have the theoretical relation $S_{f} = \sin(2\beta)$, where $S_{f}$ is a CP violation parameter.
The goal of this project is to provide an estimate of the statistical precision in $\beta$ measurements at the proposed FCC-ee using toy Monte Carlo simulations.
This is the code repository for the project, including the toy Monte Carlo simulation framework and other related analyses.


## Toy Monte Carlo Simulation

The file simulation.py performs the simulation and writes the configuration and simulation results. The file analysis.py then reads the output from the simulation to produce plots.


## Single Pseudo-Experiment Simulation

The file single.py performs a single pseudo-experiment and generates truth-level PDF plots and the iminuit output for the run.


## Fisher Information Approach

An alternative approach to the project is predicting the precision anlytically via the Fisher information, which is beyond the scope of this project. This part is included here only for demonstration purposes.

The files fisher_information_integration.py gives the exact per-event Fisher information of the CP violation parameter $S_{f}$ in the statistical model. The file fisher_information_analysis.py then computes the predicted precision in $S_{f}$ under significant simplification assumptions.
