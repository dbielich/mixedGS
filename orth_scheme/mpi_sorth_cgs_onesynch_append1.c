#include "../mixedGS.h"

int mpi_sorth_cgs_onesynch_append1(int mloc, int i, float *A, int lda, float *r, float *h, MPI_Comm mpi_comm){

	if( i == 0){

		// Do nothing in Single Precision

	} else {

		cblas_sgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1,
			0.0e0, &(r[0]), 1);
		MPI_Allreduce( MPI_IN_PLACE, &(r[0]), i, MPI_FLOAT, MPI_SUM, mpi_comm);
		cblas_sgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(r[0]), 1,
			1.0e0, &(A[i*mloc]), 1);
	
	}

	return 0;

}
