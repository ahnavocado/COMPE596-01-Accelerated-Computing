#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sched.h>

#define ORDER 1000
#define AVAL 3.0
#define BVAL 5.0

int main(int argc, char *argv[])
{
	int Ndim, Pdim, Mdim;
	int i,j,k;
	int tmp;
	double *A, *B, *C;
	double start_time, run_time, dN,mflops;
	Ndim = ORDER;
	Pdim = ORDER;
	Mdim = ORDER;

	/* A[N][P], B[P][M], C[N][M] */
	A = (double *)malloc(Ndim*Pdim*sizeof(double));
	B = (double *)malloc(Pdim*Mdim*sizeof(double));
	C = (double *)malloc(Ndim*Mdim*sizeof(double));

	/* Initialize matrices */
	for (i=0; i<Ndim; i++)
		for (j=0; j<Pdim; j++)
			*(A+(i*Ndim+j)) = AVAL;
	for (i=0; i<Pdim; i++)
		for (j=0; j<Mdim; j++)
			*(B+(i*Pdim+j)) = BVAL;
	for (i=0; i<Ndim; i++)
		for (j=0; j<Mdim; j++)
			*(C+(i*Ndim+j)) = 0.0;





	for (int num_threads = 1; num_threads <= 1024; num_threads *= 2) {
		omp_set_num_threads(num_threads);
		start_time = omp_get_wtime();
		#pragma omp parallel for private(tmp, i, j, k)
		for (i=0; i<Ndim; i++) {
			for (j=0; j<Mdim; j++) {
				tmp = 0.0;
				for(k=0; k<Pdim; k++) {
					/* C(i,j) = sum(over k) A(i,k) * B(k,j) */
					tmp += *(A+(i*Ndim+k)) * *(B+(k*Pdim+j));
				}
				*(C+(i*Ndim+j)) = tmp;
			}
		}




		run_time = omp_get_wtime() - start_time;


		dN = (double)ORDER;
		mflops = 2.0 * dN * dN * dN/(1000000.0 * run_time);
    printf("%d threads: %f seconds, %f MFLOPS\n", num_threads, run_time, mflops);
	}
	return 0;
}
