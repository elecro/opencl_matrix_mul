#include "matrix.hpp"
#include "operations.hpp"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>

#include <chrono>

struct Measurement
{
    double cpu;
    double gpu;
    double transposed;
    double dot;
    double float4;
    double constant;
};

static Matrix measure(Operations& op, Matrix& lhs, Matrix& rhs, double* spentTime)
{
    using namespace std::chrono;

    steady_clock::time_point start = std::chrono::steady_clock::now();
    Matrix matrix = op.multiply(lhs, rhs);
    steady_clock::time_point end = std::chrono::steady_clock::now();

    duration<double> span = duration_cast<std::chrono::duration<double> > (end - start);

    *spentTime = span.count();
    return matrix;
}

static Measurement measureMultiply(Matrix& lhs, Matrix& rhs)
{
    Measurement result;

    Operations* gpu = new GpuOperations();
    Matrix gpuMatrix = measure(*gpu, lhs, rhs, &result.gpu);
    delete gpu;

    Operations* transposed = new TransposedGpuOperations();
    Matrix transposedMatrix = measure(*transposed, lhs, rhs, &result.transposed);
    delete transposed;

    Operations* dot = new DotGpuOperations();
    Matrix dotMatrix = measure(*dot, lhs, rhs, &result.dot);
    delete dot;

    Operations* float4 = new Float4GpuOperations();
    Matrix float4Matrix = measure(*float4, lhs, rhs, &result.float4);
    delete float4;

    Operations* constant = new ConstantGpuOperations();
    Matrix constantMatrix = measure(*constant, lhs, rhs, &result.constant);
    delete constant;

    Operations* cpu = new CpuOperations();
    Matrix cpuMatrix = measure(*cpu, lhs, rhs, &result.cpu);
    delete cpu;

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

    if (cpuMatrix != dotMatrix)
    {
        printf("Dot Matrix mismatch\n");

        print(dotMatrix);
        printf("\nCPU Matrix:\n");
        print(cpuMatrix);
        printf("\n\n");

        printf("\n\nLHS\n");
        print(lhs);
        printf("\nRHS\n");
        print(rhs);
        printf("\n");
    }

    if (cpuMatrix != float4Matrix)
    {
        printf("Float4 Matrix mismatch\n");
    }

    return result;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printf("Usage: %s <matrix size> [count]\n", argv[0]);
        return -1;
    }

    int count = 1;
    if (argc > 2)
    {
        count = (int)strtol(argv[2], NULL, 10) ? : 1;
    }

    bool useCSVOutput = false;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp("--csv", argv[i]) == 0)
        {
            useCSVOutput = true;
            break;
        }
    }

    int size = atoi(argv[1]);
    srand (1);
//    srand (time(NULL));

    int width = size;
    int height = size;
    Matrix lhs = Matrix::random(width, height, 4);
    Matrix rhs = lhs.transpose();

    while (count-- > 0)
    {
        Measurement result = measureMultiply(lhs, rhs);

        if (useCSVOutput)
        {
            printf("%d;%d; %.4f;%.4f;%.4f;%.4f;%.4f\n",
                    width,
                    height,
                    result.cpu,
                    result.gpu,
                    result.transposed,
                    result.dot,
                    result.float4,
                    result.constant);
        }
        else
        {
            printf("%dx%d CPU: %.4f \n", width, height, result.cpu);
            printf("%dx%d GPU: %.4f \n", width, height, result.gpu);
            printf("%dx%d TPU: %.4f \n", width, height, result.transposed);
            printf("%dx%d DPU: %.4f \n", width, height, result.dot);
            printf("%dx%d 4PU: %.4f \n", width, height, result.float4);
            printf("%dx%d   1: %.4f \n", width, height, result.constant);
            printf("\n");
        }
    }

    return 0;
}
