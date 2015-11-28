#include "matrix.hpp"

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <chrono>

struct Measurement
{
    double cpu;
    double gpu;
};

static Measurement measureMultiply(Matrix& lhs, Matrix& rhs)
{
    using namespace std::chrono;

    Operations* gpu = new GpuOperations();

    steady_clock::time_point gpuStart = std::chrono::steady_clock::now();
    try {
        Matrix gpuMatrix = gpu->multiply(lhs, rhs);
    } catch (char const* e) {
        fprintf(stderr, "%s\n", e);
        throw e;
    }
    steady_clock::time_point gpuEnd = std::chrono::steady_clock::now();

    Operations* cpu = new CpuOperations();

    steady_clock::time_point cpuStart = steady_clock::now();
//    Matrix cpuMatrix = cpu->multiply(lhs, rhs);
    steady_clock::time_point cpuEnd = steady_clock::now();

    duration<double> cpuSpan = duration_cast<std::chrono::duration<double> >(cpuEnd - cpuStart);
    duration<double> gpuSpan = duration_cast<std::chrono::duration<double> >(gpuEnd - gpuStart);

/*
    if (cpuMatrix != gpuMatrix)
    {
        printf("Matrix mismatch\n");
    }
*/

    delete gpu;
    delete cpu;

    return { cpuSpan.count(), gpuSpan.count() };
}

int main()
{
    srand (time(NULL));

    std::vector<Measurement> resultSet;

    for (int i = 1; i < 41; i++)
    {
        int width = i * i;
        int height = i * i;
        Matrix lhs = Matrix::random(width, height, 10);
        Matrix rhs = lhs.transpose();

        Measurement result = measureMultiply(lhs, rhs);

        /*
        printf("%dx%d CPU: %.4f \n", width, height, result.cpu);
        printf("%dx%d GPU: %.4f \n", width, height, result.gpu);
        */
        fprintf(stderr, "%d;%d;%d;%.4f;%.4f;\n", i, width, height, result.cpu * 1000, result.gpu * 1000);
        //printf("%dx%d GPU: %.4f \n", width, height, result.gpu);

        resultSet.push_back(result);
    }

    return 0;
}
