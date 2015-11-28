__kernel void matrix_mul(__global float* A, __global float* B, __global float* C, __const int width_A, __const int width_B)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    float lhs;
    float rhs;
    float result = 0;
    for (int i = 0; i < width_A; i++)
    {
        lhs = A[y * width_A + i];
        rhs = B[i * width_B + x];
        result += dot(lhs, rhs);
    }

    C[y * width_B + x] = result;
}

