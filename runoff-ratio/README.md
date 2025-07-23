# Runoff Ratio
This directory contains a few notebooks and some Python files for calculating the runoff ratio. A quick rundown of each and what they do:
- Runoff Alternative Implementation is the original algorithm used to determine runoff ratio, but doesn't make any plots. You should use this to check your work if you end up changing any of the mechanics behind the calculations (such as picking different seasons)
- Runoff Loop Debugger is simply the main loop of the other files but with a lot more debugging output and split off into its own file so that it's easier to troubleshoot per-region bugs.
- Runoff to Precipitation Ratio is a notebook that contains the core of the plotting code and an abandoned attempt to plot them all on the same figure, which you can most likely ignore safely.
- `runoff_ratio.py` is the plotting code and the algorithm split off so that it can be run more quickly with GNU Parallel.
- `runoff_ratio_djfm.py` calculates the runoff to precipitation ratio for every model year for the period Dec-Jan-Feb-Mar.
- `runoff_ratio_mjja.py` calculates the runoff to precipitation ratio for every model year for the period May-Jun-Jul-Aug.

You can run `make summer` or `make winter` to get calculations for the `runoff_ratio`s that you want, see the `Makefile` for more information 
