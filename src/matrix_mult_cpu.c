#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 256  // Matrix size (NxN). Adjust for demo!

void matmul(const float* A, const float* B, float* C, int n) {
    for(int i=0; i<n; ++i) {
        for(int j=0; j<n; ++j) {
            float sum = 0.0f;
            for(int k=0; k<n; ++k)
                sum += A[i*n + k] * B[k*n + j];
            C[i*n + j] = sum;
        }
    }
}

void fill_rand(float* M, int n) {
    for(int i=0; i<n*n; ++i)
        M[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

int main() {
    srand((unsigned int)time(NULL));

    float* A = malloc(N*N*sizeof(float));
    float* B = malloc(N*N*sizeof(float));
    float* C = malloc(N*N*sizeof(float));

    fill_rand(A, N);
    fill_rand(B, N);

    clock_t t0 = clock();
    matmul(A, B, C, N);
    clock_t t1 = clock();

    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;

    printf("Matrix multiplication (CPU): %dx%d x %dx%d\n", N, N, N, N);
    printf("Elapsed time: %.4f seconds\n", elapsed);

    // Optional: print a few values to show result
    printf("C[0][0]=%.3f  C[N/2][N/2]=%.3f  C[N-1][N-1]=%.3f\n",
           C[0], C[(N/2)*N + (N/2)], C[N*N-1]);

    free(A);
    free(B);
    free(C);
    return 0;
}
