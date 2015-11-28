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
    double dot;
    double float4;
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

    return result;
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
        printf("%dx%d DPU: %.4f \n", width, height, result.dot);
        printf("%dx%d 4PU: %.4f \n", width, height, result.float4);
        printf("\n");
        resultSet.push_back(result);
    }

    return 0;
}
