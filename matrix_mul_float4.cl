__kernel void matrix_mul(__global float* A, __global float* B, __global float* C, int width_A, int width_B)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    float result = 0;
    float4 lhs;
    float4 rhs;
    for (int i = 0; i < width_A; i+=4)
    {
        lhs = (float4)(A[y * width_A + i + 0], A[y * width_A + i + 1], A[y * width_A + i + 2], A[y * width_A + i + 3]);
        rhs = (float4)(B[i * width_B + x], B[(i + 1) * width_B + x], B[(i + 2) * width_B + x], B[(i + 3) * width_B + x]);
        result += dot(lhs, rhs);
    }

    C[y * width_B + x] = result;
}
