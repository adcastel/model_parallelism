



/home/adcastel/opt/mpich/bin/mpicc -O3 dnn.c -o dnn  -I/opt/intel/compilers_and_libraries/linux/mkl/include/  /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_intel_lp64.a -L/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64 -L/opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/lib/intel64_lin/ -lmkl_intel_thread -lmkl_core -liomp5 -lpthread  -lmkl_scalapack_lp64 -lm -DTIMER -DVGG16 -fopenmp -DSTATIC -DSUMMARY
