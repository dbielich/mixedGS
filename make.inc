LOADER = mpif90
CC = mpicc -O3 -Wall

INCLUDE = /Users/daniel_bielich/Documents/lapack/LAPACKE/include/ -I/Users/daniel_bielich/Documents/lapack/CBLAS/include/
LIB =  /Users/daniel_bielich/Documents/lapack/liblapack.a /Users/daniel_bielich/Documents/lapack/liblapacke.a /Users/daniel_bielich/Documents/lapack/librefblas.a /Users/daniel_bielich/Documents/lapack/libcblas.a  
