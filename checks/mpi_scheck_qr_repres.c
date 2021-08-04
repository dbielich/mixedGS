#include "../mixedGS.h"

int mpi_scheck_qr_repres( MPI_Comm mpi_comm, float *norm_repres, int m, int n, float *A, int lda, float *Q, int ldq, float *R, int ldr ){

	float *work;
	int ii, jj;

	work  = (float *) malloc(m * n * sizeof(float));
	LAPACKE_slacpy_work( LAPACK_COL_MAJOR, 'A', m, n, Q, ldq, work, m );
	cblas_strmm( CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, m, n, (1.0e+00), R, ldr, work, m );
 	for(ii = 0; ii < m; ii++) for(jj = 0; jj < n; jj++) work[ ii+jj*m ] -= A[ ii+jj*lda ];
	(*norm_repres) = LAPACKE_slange_work( LAPACK_COL_MAJOR, 'F', m, n, work, m, NULL );
	free( work );

	(*norm_repres) = (*norm_repres) * (*norm_repres);
	MPI_Allreduce( MPI_IN_PLACE, norm_repres, 1, MPI_FLOAT, MPI_SUM, mpi_comm);
	(*norm_repres) = sqrt((*norm_repres));

	return 0;
}

