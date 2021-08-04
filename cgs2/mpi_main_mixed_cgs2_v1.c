#include "../mixedGS.h"

int main(int argc, char ** argv) {

	double *dA, *dQ, *dR, *dwork;
	float *sA, *sQ, *sR, *swork;
	double dnorm_orth=0.0e+00, dnorm_repres=0.0e+00;
	float snorm_orth=0.0e+00, snorm_repres=0.0e+00;
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
	for(i = 0; i < lda * n; i++) *(sA + i) = ( (float) *(dA + i) );

// 	for(i = 0; i < lda * n; i++)
//		*(sA + i) = (float)rand() / (float)(RAND_MAX) - 0.5e+00;
//	for(i = 0; i < lda * n; i++) *(dA + i) = *(sA + i);

	LAPACKE_dlacpy_work( LAPACK_COL_MAJOR, 'A', mloc, n, dA, lda, dQ, ldq );
	LAPACKE_slacpy_work( LAPACK_COL_MAJOR, 'A', mloc, n, sA, lda, sQ, ldq );

	MPI_Barrier( MPI_COMM_WORLD );
	elapse = -MPI_Wtime();

	for( j=0; j<n; j++ ){

		// Double Precision CGS2
        	if( j == 0){
        
        		dR[0] = cblas_ddot(mloc,dQ,1,dQ,1);
        		MPI_Allreduce( MPI_IN_PLACE, &(dR[0]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        		dR[0] = sqrt(dR[0]);
        		cblas_dscal(mloc,1/dR[0],dQ,1);
        
        	} else {
        
        		cblas_dgemv(CblasColMajor, CblasTrans, mloc, j, 1.0e0,  dQ, mloc, &(dQ[j*mloc]), 1,
        			0.0e0, &(dR[j*ldr]), 1);
        		MPI_Allreduce( MPI_IN_PLACE, &(dR[j*ldr]), j, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        		cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, j, (-1.0e0), dQ, mloc, &(dR[j*ldr]), 1,
        			1.0e0, &(dQ[j*mloc]), 1);
        
        		cblas_dgemv(CblasColMajor, CblasTrans, mloc, j, 1.0e0, dQ, mloc, &(dQ[j*mloc]), 1,
        			0.0e0, &(dwork[0]), 1);
        		MPI_Allreduce( MPI_IN_PLACE, &(dwork[0]), j, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        		cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, j, (-1.0e0), dQ, mloc, &(dwork[0]), 1,
        			1.0e0, &(dQ[j*mloc]), 1);
        
        		for(i=0;i<j;i++) dR[i+j*ldr] += dwork[j];
        	
        		dR[j+j*ldr] = cblas_ddot(mloc,&(dQ[j*mloc]),1,&(dQ[j*mloc]),1);
        		MPI_Allreduce( MPI_IN_PLACE, &(dR[j+j*ldr]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        		dR[j+j*ldr] = sqrt(dR[j+j*ldr]);
        
        		cblas_dscal(mloc,1/dR[j+j*ldr],&(dQ[j*mloc]),1);
        
        	}


		// Single Precision CGS2
		if( j == 0){
        
        		sR[0] = cblas_sdot(mloc,sQ,1,sQ,1);
        		MPI_Allreduce( MPI_IN_PLACE, &(sR[0]), 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        		sR[0] = sqrt(sR[0]);
        		cblas_sscal(mloc,1/sR[0],sQ,1);
        
        	} else {
        
        		cblas_sgemv(CblasColMajor, CblasTrans, mloc, j, 1.0e0,  sQ, mloc, &(sQ[j*mloc]), 1,
        			0.0e0, &(sR[j*ldr]), 1);
        		MPI_Allreduce( MPI_IN_PLACE, &(sR[j*ldr]), j, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        		cblas_sgemv(CblasColMajor, CblasNoTrans, mloc, j, (-1.0e0), sQ, mloc, &(sR[j*ldr]), 1,
        			1.0e0, &(sQ[j*mloc]), 1);
        
        		cblas_sgemv(CblasColMajor, CblasTrans, mloc, j, 1.0e0, sQ, mloc, &(sQ[j*mloc]), 1,
        			0.0e0, &(swork[0]), 1);
        		MPI_Allreduce( MPI_IN_PLACE, &(swork[0]), j, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        		cblas_sgemv(CblasColMajor, CblasNoTrans, mloc, j, (-1.0e0), sQ, mloc, &(swork[0]), 1,
        			1.0e0, &(sQ[j*mloc]), 1);
        
        		for(i=0;i<j;i++) sR[i+j*ldr] += swork[j];
        	
        		sR[j+j*ldr] = cblas_sdot(mloc,&(sQ[j*mloc]),1,&(sQ[j*mloc]),1);
        		MPI_Allreduce( MPI_IN_PLACE, &(sR[j+j*ldr]), 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        		sR[j+j*ldr] = sqrt(sR[j+j*ldr]);
        
        		cblas_sscal(mloc,1/sR[j+j*ldr],&(sQ[j*mloc]),1);
        
        	}




	}


//	for(i=0;i<n;i++){ for(j=0;j<n;j++){ printf("%+3.2e, ",dR[i+j*ldr]); } printf("\n");}
//	printf("\n");
//	for(i=0;i<n;i++){ for(j=0;j<n;j++){ printf("%+3.2e, ",sR[i+j*ldr]); } printf("\n");}

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
