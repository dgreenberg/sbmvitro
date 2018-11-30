# sbmvitro
This repository contains tools for fitting the sequential binding model (SBM) to isothermal titration calorimetry, fluorescence spectroscopy and stopped flow data.

It is written in pure python, but if [Numba](http://numba.pydata.org/) is available the derivative and Jacobian functions used by the ODE solver for stopped-flow data will be compiled on import.

For details of the model and fitting procedures, see the [bioRxiv paper](https://www.biorxiv.org/content/early/2018/11/29/479055). Examples showing how to use this software can be found in `global_fit.py`, and examples of the required data formats can be found in the paper's supplementary material. Further documentation of the API will be available in the near future.
