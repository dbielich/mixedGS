#include "../mixedGS.h"

int mpi_sorth_cgs2_threesynch_append1(int mloc, int i, float *A, int lda, float *r, float *h, MPI_Comm mpi_comm){

	int j;

	if( i == 0){

		r[0] = cblas_sdot(mloc,A,1,A,1);
		MPI_Allreduce( MPI_IN_PLACE, &(r[0]), 1, MPI_FLOAT, MPI_SUM, mpi_comm);
		r[0] = sqrt(r[0]);
		cblas_sscal(mloc,1/r[0],A,1);

	} else {

		cblas_sgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1,
			0.0e0, &(r[0]), 1);
		MPI_Allreduce( MPI_IN_PLACE, &(r[0]), i, MPI_FLOAT, MPI_SUM, mpi_comm);
		cblas_sgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(r[0]), 1,
			1.0e0, &(A[i*mloc]), 1);

		cblas_sgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1,
			0.0e0, &(h[0]), 1);
		MPI_Allreduce( MPI_IN_PLACE, &(h[0]), i, MPI_FLOAT, MPI_SUM, mpi_comm);
		cblas_sgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(h[0]), 1,
			1.0e0, &(A[i*mloc]), 1);

		for(j=0;j<i;j++) r[j] += h[j];
	
		r[i] = cblas_sdot(mloc,&(A[i*mloc]),1,&(A[i*mloc]),1);
		MPI_Allreduce( MPI_IN_PLACE, &(r[i]), 1, MPI_FLOAT, MPI_SUM, mpi_comm);
		r[i] = sqrt(r[i]);

		cblas_sscal(mloc,1/r[i],&(A[i*mloc]),1);

	}

	return 0;

}
