#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_THREADS 2  // OpenMP 스레드 개수

// 적분할 함수 정의
double f(double x) {
    return acos(cos(x) / (1 + 2 * cos(x)));
}

// 병렬 Simpson 적분 구현
double simpsons_parallel(double a, double b, int N) {
    if (N % 2 != 0) N++;  // 짝수로 맞추기
    double h = (b - a) / N;
    double sum_odd = 0.0, sum_even = 0.0;

    // 실행 시간 측정 시작
    double start_time = omp_get_wtime();

    // OpenMP 병렬 for문: Simpson 적분의 핵심
    #pragma omp parallel for reduction(+:sum_odd, sum_even) num_threads(NUM_THREADS)
    for (int i = 1; i < N; i++) {
        double x = a + i * h;
        if (i % 2 == 0)
            sum_even += f(x);  // 짝수 인덱스 가중치 2 적용
        else
            sum_odd += f(x);   // 홀수 인덱스 가중치 4 적용
    }

    double integral = (h / 3.0) * (f(a) + 4.0 * sum_odd + 2.0 * sum_even + f(b));

    double end_time = omp_get_wtime();

    printf("Execution Time: %f seconds\n", end_time - start_time);
    return integral;
}

int main() {
    double a = 0.0, b = M_PI_2;  // 적분 구간 [0, π/2]
    int N = 10000000;  // 구간 개수 (큰 값으로 설정)

    double exact_solution = (5.0 * M_PI * M_PI) / 24.0;  // 주어진 정확한 해
    double result = simpsons_parallel(a, b, N);  // 적분 수행

    // 결과 출력
    printf("Approximated Integral: %.15f\n", result);
    printf("Exact Solution: %.15f\n", exact_solution);
    printf("Absolute Error: %.15e\n", fabs(result - exact_solution));

    return 0;
}
