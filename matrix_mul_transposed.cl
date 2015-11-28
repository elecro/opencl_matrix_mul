__kernel void matrix_transpose(__global float* src, __global float* dst)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    dst[x * height + y] = src[y * width + x];
}

__kernel void matrix_mul(__global float* A, __global float* B, __global float* C, __const int width_A, __const int height_B)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    float result = 0;
    for (int i = 0; i < width_A; i++)
    {
        result += A[y * width_A + i] * B[x * width_A + i];
    }

    C[y * height_B + x] = result;
}
