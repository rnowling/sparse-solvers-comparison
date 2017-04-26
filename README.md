# sparse-solvers-comparison
Comparison of Sparse Iterative Solvers for GPUs

## Scipy
To run the GMRES solver:

```
$ cd scipy
$ python scipy_gmres.py --matrix-fl ../test_data/small_grid/A_matrix.mtx --input-vec ../test_data/small_grid/b_vector.txt
```

## CUSP Library
[CUSP](http://cusplibrary.github.io/) is a library for solving sparse linear systems built on Thrust and CUDA.

To run the GMRES solver:

```
$ cd cusp_gmres
$ make
$ ./cusp_gmres diag ../test_data/small_grid/A_matrix.mtx ../test_data/small_grid/b_vector.txt x_vector.txt
```

To run the CG solver:

```
$ cd cusp_cg
$ make
$ ./cusp_cg diag ../test_data/small_grid/A_matrix.mtx ../test_data/small_grid/b_vector.txt x_vector.txt
```

## ViennaCL Library
[ViennaCL](http://viennacl.sourceforge.net/) is a library for solving sparse linear systems which supports OpenCL, CUDA, and Intel MIC accelerators.

To run the GMRES solver:

```
$ cd viennacl
$ make
$ ./viennacl diag ../test_data/small_grid/A_matrix.mtx ../test_data/small_grid/b_vector.txt x_vector.txt
```
