#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define PI 3.14159265358979323846

// 적분할 함수 f(x) = arccos( cos(x) / (1 + 2cos(x)) )
double f(double x) {
    return acos(cos(x) / (1 + 2 * cos(x)));
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <num_threads> <num_intervals>\n", argv[0]);
        return 1;
    }

    // 사용자 입력으로 스레드 개수 및 n 설정
    int Nthrds;
    unsigned long long n;
    sscanf(argv[1], "%d", &Nthrds);
    sscanf(argv[2], "%llu", &n);

    // n이 짝수가 아닌 경우 오류 처리
    if (n % 2 != 0) {
        printf("Error: Number of intervals (n) must be even.\n");
        return 1;
    }

    printf("Requesting %d threads\n", Nthrds);
    printf("Using %llu intervals\n", n);
    
    // OpenMP 설정
    omp_set_dynamic(0);
    omp_set_num_threads(Nthrds);

    // 적분 구간 및 변수 설정
    const double a = 0.0, b = PI / 2.0;
    const double h = (b - a) / n; // 구간 크기

    double S = 0.0; // 최종 적분 결과

    // 실행 시간 측정 시작
    double start_time = omp_get_wtime();

    // Simpson's Rule 적용 (병렬 합산)
    #pragma omp parallel for reduction(+:S)
    for (unsigned long long j = 1; j <= n / 2; ++j) {
        double x0 = a + (2 * j - 2) * h;
        double x1 = a + (2 * j - 1) * h;
        double x2 = a + (2 * j) * h;
        S += f(x0) + 4 * f(x1) + f(x2);
    }

    // 최종 적분 값 계산
    S *= h / 3.0;

    // 실행 시간 측정 종료
    double end_time = omp_get_wtime();

    // 정확한 해
    double exact_solution = (5.0 * PI * PI) / 24.0;
    double error = fabs(S - exact_solution);

    // 결과 출력
    printf("Approximated integral result: %.15lf\n", S);
    printf("Exact integral solution: %.15lf\n", exact_solution);
    printf("Numerical error: %.15le\n", error);
    printf("Execution time: %lf seconds\n", end_time - start_time);

    return 0;
}

