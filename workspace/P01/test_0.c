#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define ORDER 1000  // 행렬 크기
#define AVAL 3.0
#define BVAL 5.0

int main(int argc, char *argv[])
{
    int Ndim = ORDER, Pdim = ORDER, Mdim = ORDER;
    int i, j, k;
    double *A, *B, *C;
    double start_time, run_time, dN, mflops;

    // 메모리 할당
    A = (double *)malloc(Ndim * Pdim * sizeof(double));
    B = (double *)malloc(Pdim * Mdim * sizeof(double));
    C = (double *)malloc(Ndim * Mdim * sizeof(double));

    // 행렬 초기화
    for (i = 0; i < Ndim; i++)
        for (j = 0; j < Pdim; j++)
            A[i * Pdim + j] = AVAL;

    for (i = 0; i < Pdim; i++)
        for (j = 0; j < Mdim; j++)
            B[i * Mdim + j] = BVAL;

    for (i = 0; i < Ndim; i++)
        for (j = 0; j < Mdim; j++)
            C[i * Mdim + j] = 0.0;

    for (int num_threads = 1; num_threads <= 1024; num_threads *= 2) {
        omp_set_num_threads(num_threads);  // 스레드 개수 설정

        start_time = omp_get_wtime();  // 실행 시간 측정 시작

        #pragma omp parallel for private(i, j, k) shared(A, B, C)
        for (i = 0; i < Ndim; i++) {
            for (j = 0; j < Mdim; j++) {
                double tmp = 0.0;  // 각 스레드마다 개별 변수 사용
                for (k = 0; k < Pdim; k++) {
                    tmp += A[i * Pdim + k] * B[k * Mdim + j];
                }
                C[i * Mdim + j] = tmp;
            }
        }

        run_time = omp_get_wtime() - start_time;  // 실행 시간 측정 종료

        // MFLOPS 계산
        dN = (double)ORDER;
        mflops = 2.0 * dN * dN * dN / (1000000.0 * run_time);
        printf("%d threads: %f seconds, %f MFLOPS\n", num_threads, run_time, mflops);
    }

    // 메모리 해제
    free(A);
    free(B);
    free(C);

    return 0;
}

