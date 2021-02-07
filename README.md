# STEMtranslator
Set of tools to translate STEM data into Cy-RSoXS compatible format. The growth classes provide functionality to generate syntheic morphologies

Current functionality
1. Translate semi-reduced 4DSTEM & HRTREM data (I vs. q and chi)
2. Monte Carlo crystal growth. Can be used to heal gaps in STEM data or create syntheic morphologies
3. Monte Carlo fiber growth. Orthorhombic or helically varying dielectric functions can be mapped to fibers
4. Small writer class to write morphologies to .hdf5, config.txt, and materialXX.txt files for Cy-RSoXS

To Do:
1. Functionality to read and reduce raw STEM data. Leverage existing libraries (FPDpy, mrc)
2. Implement Eliot's Monte Carlo alignment/annealing code
3. Poisson disk-like placing of fibers, so they are self avoiding
