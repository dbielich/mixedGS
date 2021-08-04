#include "../mixedGS.h"

int main(int argc, char ** argv) {

	double *dA, *dQ, *dR, *dwork;
	float *sA, *sQ, *sR, *swork;
	double dnorm_orth=0.0e+00, dnorm_repres=0.0e+00;
	float snorm_orth=0.0e+00, snorm_repres=0.0e+00;
	double elapse, nrmA;
	int lda, ldq, ldr;
	int m, n, i, j, testing;
	int my_rank, pool_size, seed, mloc;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &pool_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    	m         = 27*pool_size;
	n         = 11;
	testing   =  0;

	for(i = 1; i < argc; i++){
		if( strcmp( *(argv + i), "-testing") == 0) {
			testing = atoi( *(argv + i + 1) );
			i++;
		}
		if( strcmp( *(argv + i), "-m") == 0) {
			m = atoi( *(argv + i + 1) );
			i++;
		}
		if( strcmp( *(argv + i), "-n") == 0) {
			n = atoi( *(argv + i + 1) );
			i++;
		}
	}

	seed = my_rank*m*m; srand(seed);
	mloc = m / pool_size; if ( my_rank < m - pool_size*mloc ) mloc++;

	lda = mloc; ldq = mloc; ldr = n;

	int stop=0; if( mloc < n ){ stop = 1; }
	MPI_Allreduce( MPI_IN_PLACE, &stop, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	if ( stop > 0 ){ printf("\n\n We need n <= mloc: mloc = %3d & n = %3d\n\n",mloc,n); MPI_Finalize(); return 0; }

	swork = (float *) malloc( 2*n * sizeof(float));
	sQ = (float *) malloc( ldq * n * sizeof(float));
	sA = (float *) malloc( lda * n * sizeof(float));
	sR = (float *) malloc( n * n * sizeof(float));

	dwork = (double *) malloc( 2*n * sizeof(double));
	dQ = (double *) malloc( ldq * n * sizeof(double));
	dA = (double *) malloc( lda * n * sizeof(double));
	dR = (double *) malloc( n * n * sizeof(double));

 	for(i = 0; i < lda * n; i++)
		*(dA + i) = (double)rand() / (double)(RAND_MAX) - 0.5e+00;

// 	for(i = 0; i < lda * n; i++)
//		*(sA + i) = (float)rand() / (float)(RAND_MAX) - 0.5e+00;

	for(i = 0; i < lda * n; i++) *(sA + i) = *(dA + i);
//	for(i = 0; i < lda * n; i++) *(dA + i) = *(sA + i);

	LAPACKE_dlacpy_work( LAPACK_COL_MAJOR, 'A', mloc, n, dA, lda, dQ, ldq );
	LAPACKE_slacpy_work( LAPACK_COL_MAJOR, 'A', mloc, n, sA, lda, sQ, ldq );

	nrmA = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'F', mloc, n, dA, lda, NULL );

	MPI_Barrier( MPI_COMM_WORLD );
	elapse = -MPI_Wtime();

	for( j=0; j<n; j++ ){

	   mpi_sorth_cgs_onesynch_append1( mloc, j, sQ, ldq, sR+j*ldr, swork, MPI_COMM_WORLD);

   	   for(i = 0; i < ldq; i++) dQ[i+j*ldq] = ( (double) sQ[i+j*ldq] );
   	   for(i = 0; i < j; i++) dR[i+j*ldr] = ( (double) sR[i+j*ldr] );

	   mpi_dorth_cgs_twosynch_append1( mloc, j, dQ, ldq, dR+j*ldr, dwork, MPI_COMM_WORLD);

   	   for(i = 0; i <= j; i++) sR[i+j*ldr] = ( (float) dR[i+j*ldr] );

	}


	for(i=0;i<n;i++){ for(j=0;j<n;j++){ printf("%+3.2e, ",dR[i+j*ldr]); } printf("\n");}
	printf("\n");
	for(i=0;i<n;i++){ for(j=0;j<n;j++){ printf("%+3.2e, ",sR[i+j*ldr]); } printf("\n");}

	MPI_Barrier( MPI_COMM_WORLD );
	elapse += MPI_Wtime();

	if ( testing ){

		if( my_rank == 0 ) printf("-------------------------------------------------------------------------\n");
		if( my_rank == 0 ) printf("          At the final step (in single precision) step we have\n");
		
		mpi_scheck_qq_orth( MPI_COMM_WORLD, &snorm_orth, mloc, n, sQ, ldq );
		if( my_rank == 0 ) printf("                %5.1e ",snorm_orth);
		mpi_scheck_qr_repres( MPI_COMM_WORLD, &snorm_repres, mloc, n, sA, lda, sQ, ldq, sR, ldr );
		if( my_rank == 0 ) printf(" %5.1e\n",snorm_repres);
		if( my_rank == 0 ) printf("-------------------------------------------------------------------------\n");

	}
	if ( testing ){

		if( my_rank == 0 ) printf("-------------------------------------------------------------------------\n");
		if( my_rank == 0 ) printf("          At the final step (in double precision) step we have\n");
		
		mpi_dcheck_qq_orth( MPI_COMM_WORLD, &dnorm_orth, mloc, n, dQ, ldq );
		if( my_rank == 0 ) printf("                %5.1e ",dnorm_orth);
		mpi_dcheck_qr_repres( MPI_COMM_WORLD, &dnorm_repres, mloc, n, dA, lda, dQ, ldq, dR, ldr );
		if( my_rank == 0 ) printf(" %5.1e\n",dnorm_repres);
		if( my_rank == 0 ) printf("-------------------------------------------------------------------------\n");

	}
	if( my_rank == 0 ) printf("%3d %6d %6d %15.8f\n", pool_size, m, n, elapse);

	free( sA );
	free( sQ );
	free( sR );
	free( swork );
	free( dA );
	free( dQ );
	free( dR );
	free( dwork );

        MPI_Finalize();
	return 0;

}
