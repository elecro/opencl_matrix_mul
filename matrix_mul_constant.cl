__kernel void matrix_mul(__global float* A, __global float* B, __global float* C, int width_A, int width_B)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    float result = 0;
    for (int i = 0; i < width_A; i++)
    {
        result += 1;
    }

    C[y * width_B + x] = result;
}
