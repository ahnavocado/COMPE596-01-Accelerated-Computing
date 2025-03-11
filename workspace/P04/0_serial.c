#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define T(i, j) (T[(i)*(N+2) + (j)])
#define T_new(i, j) (T_new[(i)*(N+2) + (j)])

void print_matrix(double *T, int N) {
    printf("T%d = [...\n", N);
    for (unsigned i = 0; i <= N + 1; i++) {
        for (unsigned j = 0; j <= N + 1; j++) {
            printf("%.6f ", T(i, j));
        }
        printf("; ...\n");
    }
    printf("] ;\n\n");
}

void jacobi_serial(double *T, int N, int max_iter) {
    int iteration = 0;
    double residual = 1e6;
    double *T_new = (double *)malloc((N + 2) * (N + 2) * sizeof(double));
    clock_t start_time = clock();

    while (residual > 1e-8 && iteration < max_iter) {
        residual = 0.0;
        for (unsigned i = 1; i <= N; i++) {
            for (unsigned j = 1; j <= N; j++) {
                T_new(i, j) = 0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));
            }
        }
        for (unsigned i = 1; i <= N; i++) {
            for (unsigned j = 1; j <= N; j++) {
                residual = MAX(fabs(T_new(i, j) - T(i, j)), residual);
                T(i, j) = T_new(i, j);
            }
        }
        iteration++;
    }
    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("& Using number of cells = %d\n", N);
    printf("& Using maximum iteration count = %d\n", max_iter);
    printf("& Serial Residual = %.9e\n", residual);
    printf("& CPU serial kernel time = %.6f Seconds\n", elapsed_time);
    print_matrix(T, N);
    
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
    
    // Initialize boundary conditions
    for (unsigned i = 0; i <= N + 1; i++) {
        for (unsigned j = 0; j <= N + 1; j++) {
            if (j == 0 || j == (N + 1))
                T(i, j) = 1.0;
            else
                T(i, j) = 0.0;
        }
    }
    
    jacobi_serial(T, N, max_iter);
    
    free(T);
    return 0;
}

