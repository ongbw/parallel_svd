This directory contains the C source code used to generate the timing results in the SIMA article,
"A distributed and Incremental SVD Algorithm for Agglomerative Data Analysis on Large Networks"

Each stand alone c code reads in data from a pre-existing hdf5 file, which can be generated from the 
provided matlab scripts.  The intel MKL library was used to provide the dgesvd function.  To compile,
you will probably need to link against 

(1) the MKL library:
    -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm -lpthread 

(2) HDF5 libary:
     -lhdf5 -lhdf5_hl 

(3) PAPI (for measuring FLOPs)
    -lpapi

