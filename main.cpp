#include "matrix.hpp"
#include "operations.hpp"

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <chrono>

struct Measurement
{
    double cpu;
    double gpu;
    double transposed;
};

static Measurement measureMultiply(Matrix& lhs, Matrix& rhs)
{
    using namespace std::chrono;

    Operations* gpu = new GpuOperations();

    steady_clock::time_point gpuStart = std::chrono::steady_clock::now();
    Matrix gpuMatrix = gpu->multiply(lhs, rhs);
    steady_clock::time_point gpuEnd = std::chrono::steady_clock::now();

    delete gpu;

    Operations* transposed = new TransposedGpuOperations();
    steady_clock::time_point transposedStart = steady_clock::now();
    Matrix transposedMatrix = transposed->multiply(lhs, rhs);
    steady_clock::time_point transposedEnd = steady_clock::now();

    delete transposed;

    Operations* cpu = new CpuOperations();

    steady_clock::time_point cpuStart = steady_clock::now();
    Matrix cpuMatrix = cpu->multiply(lhs, rhs);
    steady_clock::time_point cpuEnd = steady_clock::now();

    duration<double> transposedSpan = duration_cast<std::chrono::duration<double> >(transposedEnd - transposedStart);
    duration<double> cpuSpan = duration_cast<std::chrono::duration<double> >(cpuEnd - cpuStart);
    duration<double> gpuSpan = duration_cast<std::chrono::duration<double> >(gpuEnd - gpuStart);

    if (cpuMatrix != gpuMatrix)
    {
        printf("GPU Matrix mismatch\n");
    }

    if (cpuMatrix != transposedMatrix)
    {
        printf("Transposed Matrix mismatch\n");

        print(transposedMatrix);
        printf("\nCPU Matrix:\n");
        print(cpuMatrix);
        printf("\n\n");

        printf("\n\nLHS\n");
        print(lhs);
        printf("\nRHS\n");
        print(rhs);
        printf("\n");

        printf("\nT RHS\n");
        Matrix t = rhs.transpose();
        print(t);
        printf("\n");

    }

    delete cpu;

    return { cpuSpan.count(), gpuSpan.count(), transposedSpan.count() };
}

int main()
{
    srand (time(NULL));

    std::vector<Measurement> resultSet;

    for (int i = 1; i < 2; i++)
    {
        int width = 3 * i;
        int height = 2 * i;
        Matrix lhs = Matrix::random(width, height, 20);
        Matrix rhs = lhs.transpose();

        Measurement result = measureMultiply(lhs, rhs);

        printf("%dx%d CPU: %.4f \n", width, height, result.cpu);
        printf("%dx%d GPU: %.4f \n", width, height, result.gpu);
        printf("%dx%d TPU: %.4f \n", width, height, result.transposed);
        //fprintf(stderr, "%d;%d;%d;%.4f;%.4f;\n", i, width, height, result.cpu * 1000, result.gpu * 1000);
        //printf("%dx%d GPU: %.4f \n", width, height, result.gpu);
        printf("\n");
        resultSet.push_back(result);
    }

    return 0;
}
