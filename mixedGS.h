#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "mpi.h"

#if !defined(USE_MKL)
#include "cblas.h"
#include "lapacke.h"
#endif


extern int mpi_sorth_cgs2_threesynch_append1(int mloc, int i, float *A, int lda, float *r, float *h, MPI_Comm mpi_comm);
extern int mpi_dorth_cgs2_threesynch_append1(int mloc, int i, double *A, int lda, double *r, double *h, MPI_Comm mpi_comm);


extern int mpi_dcheck_qr_repres( MPI_Comm mpi_comm, double *norm_repres, int m, int n, double *A, int lda, double *Q, int ldq, double *R, int ldr );
extern int mpi_dcheck_qq_orth( MPI_Comm mpi_comm, double *orthlevel, int mloc, int n, double *Q, int ldq  );

extern int mpi_scheck_qq_orth( MPI_Comm mpi_comm, float *orthlevel, int mloc, int n, float *Q, int ldq  );
extern int mpi_scheck_qr_repres( MPI_Comm mpi_comm, float *norm_repres, int m, int n, float *A, int lda, float *Q, int ldq, float *R, int ldr );

extern int mpi_sorth_cgs_onesynch_append1(int mloc, int i, float *A, int lda, float *r, float *h, MPI_Comm mpi_comm);
extern int mpi_dorth_cgs_twosynch_append1(int mloc, int i, double *A, int lda, double *r, double *h, MPI_Comm mpi_comm);






