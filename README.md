# sparse-solvers-comparison
Comparison of Sparse Iterative Solvers for GPUs

## CUSP Library
[CUSP](http://cusplibrary.github.io/) is a library for solving sparse linear systems built on Thrust and CUDA.

To run the GMRES solver:

```
$ cd cusp_gmres
$ make
$ ./cusp_gmres ../test_data/small_grid/A_matrix.mtx ../test_data/small_grid/b_vector.txt
```

To run the CG solver:

```
$ cd cusp_cg
$ make
$ ./cusp_cg ../test_data/small_grid/A_matrix.mtx ../test_data/small_grid/b_vector.txt
```