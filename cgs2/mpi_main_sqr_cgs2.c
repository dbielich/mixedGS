#include "../mixedGS.h"

int main(int argc, char ** argv) {

	float *A, *Q, *R, *work;
	float norm_orth=0.0e+00, norm_repres=0.0e+00, nrmA;
	double elapse;
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

	work = (float *) malloc( 2*n * sizeof(float));

	Q = (float *) malloc( ldq * n * sizeof(float));
	A = (float *) malloc( lda * n * sizeof(float));
	R = (float *) malloc( n * n * sizeof(float));

 	for(i = 0; i < lda * n; i++)
		*(A + i) = (float)rand() / (float)(RAND_MAX) - 0.5e+00;

	LAPACKE_slacpy_work( LAPACK_COL_MAJOR, 'A', mloc, n, A, lda, Q, ldq );
	nrmA = LAPACKE_slange_work( LAPACK_COL_MAJOR, 'F', mloc, n, A, lda, NULL );

	MPI_Barrier( MPI_COMM_WORLD );
	elapse = -MPI_Wtime();

	for( j=0; j<n; j++ ){
	   mpi_sorth_cgs2_threesynch_append1( mloc, j, Q, mloc, R+j*ldr, work, MPI_COMM_WORLD);
	}


	MPI_Barrier( MPI_COMM_WORLD );
	elapse += MPI_Wtime();

	if ( testing ){

		if( my_rank == 0 ) printf("-----------------------------------------------------\n");
		if( my_rank == 0 ) printf("          At the final step step we have\n");
		
		mpi_scheck_qq_orth( MPI_COMM_WORLD, &norm_orth, mloc, n, Q, ldq );
		if( my_rank == 0 ) printf("                %5.1e ",norm_orth);
		mpi_scheck_qr_repres( MPI_COMM_WORLD, &norm_repres, mloc, n, A, lda, Q, ldq, R, ldr );
		if( my_rank == 0 ) printf(" %5.1e\n",norm_repres);
		if( my_rank == 0 ) printf("-----------------------------------------------------\n");

	}
	if( my_rank == 0 ) printf("%3d %6d %6d %15.8f\n", pool_size, m, n, elapse);

	free( A );
	free( Q );
	free( R );
	free( work );

        MPI_Finalize();
	return 0;

}
