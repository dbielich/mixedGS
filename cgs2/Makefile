
include ../make.inc

#all: xmpi_main_dqr_cgs2.exe
#all: xmpi_main_sqr_cgs2.exe
#all: xmpi_main_sdqr_cgs2.exe

all: xmpi_main_mixed_cgs2.exe

xmpi_main_dqr_cgs2.exe: mpi_main_dqr_cgs2.o ../orth_scheme/mpi_dorth_cgs2_threesynch_append1.o ../checks/mpi_dcheck_qq_orth.o ../checks/mpi_dcheck_qr_repres.o
	$(LOADER) -o $@ mpi_main_dqr_cgs2.o ../orth_scheme/mpi_dorth_cgs2_threesynch_append1.o ../checks/mpi_dcheck_qq_orth.o ../checks/mpi_dcheck_qr_repres.o $(LIB)

xmpi_main_sqr_cgs2.exe: mpi_main_sqr_cgs2.o ../orth_scheme/mpi_sorth_cgs2_threesynch_append1.o ../checks/mpi_scheck_qq_orth.o ../checks/mpi_scheck_qr_repres.o
	$(LOADER) -o $@ mpi_main_sqr_cgs2.o ../orth_scheme/mpi_sorth_cgs2_threesynch_append1.o ../checks/mpi_scheck_qq_orth.o ../checks/mpi_scheck_qr_repres.o $(LIB)

xmpi_main_sdqr_cgs2.exe: mpi_main_sdqr_cgs2.o ../orth_scheme/mpi_sorth_cgs_onesynch_append1.o ../orth_scheme/mpi_dorth_cgs_twosynch_append1.o ../checks/mpi_scheck_qq_orth.o ../checks/mpi_scheck_qr_repres.o ../checks/mpi_dcheck_qq_orth.o ../checks/mpi_dcheck_qr_repres.o
	 $(LOADER) -o $@ mpi_main_sdqr_cgs2.o ../orth_scheme/mpi_sorth_cgs_onesynch_append1.o ../orth_scheme/mpi_dorth_cgs_twosynch_append1.o ../checks/mpi_scheck_qq_orth.o ../checks/mpi_scheck_qr_repres.o ../checks/mpi_dcheck_qq_orth.o ../checks/mpi_dcheck_qr_repres.o $(LIB)

xmpi_main_mixed_cgs2.exe: mpi_main_mixed_cgs2.o ../checks/mpi_dcheck_qq_orth.o ../checks/mpi_dcheck_qr_repres.o ../checks/mpi_scheck_qq_orth.o ../checks/mpi_scheck_qr_repres.o
	  $(LOADER) -o $@ mpi_main_mixed_cgs2.o ../checks/mpi_dcheck_qq_orth.o ../checks/mpi_dcheck_qr_repres.o ../checks/mpi_scheck_qq_orth.o ../checks/mpi_scheck_qr_repres.o $(LIB)

.c.o:
	$(CC) $(OPTS) -I$(INCLUDE) -c -o $@ $<

clean:
	rm -f *o *exe 
