#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define IDX(i, j, N) ((i) * (N+2) + (j))

void jacobi_cpu_parallel(double *T, int N, int max_iter) {
    int iteration = 0;
    double residual = 1e6;
    double *T_new = (double *)malloc((N + 2) * (N + 2) * sizeof(double));
    double start_time = omp_get_wtime();
    int num_threads = 1;

    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    
    while (residual > 1e-8 && iteration < max_iter) {
        residual = 0.0;
        
        #pragma omp parallel for collapse(2)
        for (unsigned i = 1; i <= N; i++) {
            for (unsigned j = 1; j <= N; j++) {
                T_new[IDX(i, j, N)] = 0.25 * (T[IDX(i+1, j, N)] + T[IDX(i-1, j, N)] +
                                             T[IDX(i, j+1, N)] + T[IDX(i, j-1, N)]);
            }
        }
        
        #pragma omp parallel for collapse(2) reduction(max:residual)
        for (unsigned i = 1; i <= N; i++) {
            for (unsigned j = 1; j <= N; j++) {
                double diff = fabs(T_new[IDX(i, j, N)] - T[IDX(i, j, N)]);
                residual = MAX(diff, residual);
                T[IDX(i, j, N)] = T_new[IDX(i, j, N)];
            }
        }
        iteration++;
    }
    
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    printf("Using number of cells = %d\n", N);
    printf("Using maximum iteration count = %d\n", max_iter);
    printf("Residual = %.9e\n", residual);
    printf("OpenMP CPU time = %.6f Sec\n", elapsed_time);
    printf("Number of OpenMP threads = %d\n", num_threads);

    free(T_new);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <num_cells> <max_iterations>\n", argv[0]);
        return 1;
    }
    
    int N = atoi(argv[1]);
    int max_iter = atoi(argv[2]);
    
    double *T = (double *)malloc((N + 2) * (N + 2) * sizeof(double));
    
    #pragma omp parallel for collapse(2)
    for (unsigned i = 0; i <= N + 1; i++) {
        for (unsigned j = 0; j <= N + 1; j++) {
            T[IDX(i, j, N)] = (j == 0 || j == (N + 1)) ? 1.0 : 0.0;
        }
    }
    
    jacobi_cpu_parallel(T, N, max_iter);
    
    free(T);
    return 0;
}

