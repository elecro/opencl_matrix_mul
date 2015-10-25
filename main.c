#include <stdio.h>
#include <CL/cl.h>

struct size
{
    int width;
    int height;
};

float* allocate_matrix(int width, int height)
{
    return (float*)malloc(sizeof(float) * width * height);
}

void fill_matrix(float* matrix, int width, int height, float value)
{
    for (int i = 0; i < width * height; i++)
    {
        matrix[i] = value;
    }
}

void print_matrix(float* A, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            printf("%.2f ", A[y * width + x]);
        }
        printf("\n");
    }
}

float* matrix_mul(float* A, float* B, struct size matrix_result, struct size matrix_A, struct size matrix_B)
{
    float* result = allocate_matrix(matrix_result.width, matrix_result.height);

    for (int i = 0; i < matrix_result.width * matrix_result.height; i++)
    {
        int x = i % matrix_result.width;
        int y = i / matrix_result.width;
        float value = 0;

        for (int j = 0; j < matrix_A.width; j++)
        {
            value += A[y * matrix_A.width + j] * B[j * matrix_B.width + x];
        }

        result[i] = value;
    }

    return result;
}

int matrix_equals(float* A, float* B, int length)
{
    for (int i = 0; i < length; i++)
    {
        if (A[i] != B[i])
        {
            return 0;
        }
    }

    return 1;
}

char* read_file(char* filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Failed to load file: %s\n", filename);
        exit(1);
    }

    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);

    fseek(fp, 0, SEEK_SET);

    char* source = (char*)malloc(source_size);
    fread(source, 1, source_size, fp);
    fclose(fp);

    source[source_size] = 0;
    return source;
}

char* query_device_info(cl_device_id device, cl_device_info value_param)
{
    size_t value_size = 0;
    clGetDeviceInfo(device, value_param, 0, NULL, &value_size);
    char* value = (char*)malloc(sizeof(char) * value_size);
    clGetDeviceInfo(device, value_param, value_size, value, NULL);

    return value;
}

int device_is_gpu(cl_device_id device)
{
    cl_device_type device_type = CL_DEVICE_TYPE_DEFAULT;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);

    return device_type == CL_DEVICE_TYPE_GPU ? 1 : 0;
}

int main(int argc, char** argv)
{
    struct size matrix_A = { 6, 11 };
    struct size matrix_B = { 1, 6 };
    struct size matrix_C = { matrix_B.width, matrix_A.height };

/*
    int s = 2048;
    struct size matrix_A = { s, s };
    struct size matrix_B = { s, s };
    struct size matrix_C = { s, s };
*/
    float* A = allocate_matrix(matrix_A.width, matrix_A.height);
    float* B = allocate_matrix(matrix_B.width, matrix_B.height);
    float* C = allocate_matrix(matrix_C.width, matrix_C.height);

    fill_matrix(A, matrix_A.width, matrix_A.height, 3);
    fill_matrix(B, matrix_B.width, matrix_B.height, 2);
    fill_matrix(C, matrix_C.width, matrix_C.height, 0);

    printf("A: \n");
    print_matrix(A, matrix_A.width, matrix_A.height);

    printf("\nB: \n");
    print_matrix(B, matrix_B.width, matrix_B.height);
    printf("\n\n");

    if (matrix_A.width != matrix_B.height)
    {
        printf("Error: matrice sizes are incorrect!\n"
               "A: (%dx%d) B: (%dx%d)\n"
               "exiting\n",
               matrix_A.width, matrix_A.height,
               matrix_B.width, matrix_B.height);

        return -1;
    }

    // Get & select device
    cl_platform_id platform_id; // We'll support only the first platfrom for now.
    clGetPlatformIDs(1, &platform_id, NULL);

    cl_device_id selected_device = NULL;

    cl_uint device_count = 0;
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &device_count);
    cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * device_count);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, device_count, devices, NULL);

    for (cl_uint j = 0; j < device_count; j++)
    {
        char* name = query_device_info(devices[j], CL_DEVICE_NAME);
        char* hw_opencl = query_device_info(devices[j], CL_DEVICE_VERSION);
        char* sw_opencl = query_device_info(devices[j], CL_DRIVER_VERSION);

        int is_gpu = device_is_gpu(devices[j]);
        printf("%d. Device name: %s (HW: %s, SW: %s, Is gpu?: %s)\n",
                j + 1,
                name,
                hw_opencl,
                sw_opencl,
                is_gpu ? "Yes" : "No");

        if (is_gpu && selected_device == NULL)
        {
            selected_device = devices[j];
            printf(" Selected device ^^\n");
        }

        free(name);
        free(hw_opencl);
        free(sw_opencl);
    }

    if (selected_device == NULL)
    {
        printf("Error: Unable to select a GPU device\n");
    }
    else
    {
        cl_int error = 0;

        // Create OpenCL context
        cl_context context = clCreateContext(NULL, 1, &selected_device, NULL, NULL, &error);
        if (!context)
        {
            printf("Error: Unable to create a context\n");
            free(devices);
            return -1;
        }

        // Create command queue
        cl_command_queue queue = clCreateCommandQueue(context, selected_device, 0, &error);
        if (!queue)
        {
            printf("Error: Unable to create a command queue\n");
            clReleaseContext(context);
            free(devices);
            return -2;
        }

        char* kernel_source = read_file("matrix_mul.cl");

        cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &error);

//        free(kernel_source);

        if (!program)
        {
            printf("Error: Unable to create the program\n");
            clReleaseContext(context);
            free(devices);
            return -3;
        }

        error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (error)
        {
            size_t len;
            char buffer[2048];
            printf("Error: Unable to build the program, error message:\n");
            clGetProgramBuildInfo(program, selected_device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
            printf("%s\n", buffer);

            clReleaseContext(context);
            free(devices);
            return -4;
        }

        cl_kernel kernel = clCreateKernel(program, "matrix_mul", &error);
        if (!kernel || error != CL_SUCCESS)
        {
            printf("Error: Unable to create the kernel\n");

            clReleaseContext(context);
            free(devices);
            return -5;
        }

        cl_mem dev_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * matrix_A.width * matrix_A.height, A, &error);
        if (error != CL_SUCCESS)
        {
            printf("Error: Unable to create matrix buffer A\n");
            clReleaseContext(context);
            free(devices);
            return -6;
        }

        cl_mem dev_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * matrix_B.width * matrix_B.height, B, &error);
        if (error != CL_SUCCESS)
        {
            printf("Error: Unable to create matrix buffer B\n");
            clReleaseContext(context);
            free(devices);
            return -6;
        }

        cl_mem dev_C = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * matrix_C.width * matrix_C.height, NULL, &error);
        if (error != CL_SUCCESS)
        {
            printf("Error: Unable to create matrix buffer C\n");
            clReleaseContext(context);
            free(devices);
            return -6;
        }

        // TODO: test buffer errors

        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&dev_A);
        error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&dev_B);
        error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&dev_C);
        error |= clSetKernelArg(kernel, 3, sizeof(int), (void*)&matrix_A.width);
        error |= clSetKernelArg(kernel, 4, sizeof(int), (void*)&matrix_B.width);

        if (error != CL_SUCCESS)
        {
            printf("Error: Unable to set kernel arguments\n");
            clReleaseContext(context);
            free(devices);
            return -6;
        }

        size_t globalWorkSize[2] = { matrix_C.width, matrix_C.height };

        error = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL /* localWorkSize */, 0, NULL, NULL);
        if (error != CL_SUCCESS)
        {
            printf("Error: Unable to execute kernel\n");
            return -7;
        }

        error = clEnqueueReadBuffer(queue, dev_C, CL_TRUE, 0, sizeof(float) * matrix_C.width * matrix_C.height, C, 0, NULL, NULL);
        if (error != CL_SUCCESS)
        {
            printf("Error: Unable to read result array\n");
            return -8;
        }

        printf("GPU result:\n");
        print_matrix(C, matrix_C.width, matrix_C.height);

        float* cpu_result = matrix_mul(A, B, matrix_C, matrix_A, matrix_B);

        if (matrix_equals(C, cpu_result, matrix_C.width * matrix_C.height) == 1)
        {
            printf("Matrix equals\n");
        }
        else
        {
            printf("Not equals\n\n");

            printf("\nCPU result:\n");
            print_matrix(cpu_result, matrix_C.width, matrix_C.height);
        }

        clReleaseMemObject(dev_A);
        clReleaseMemObject(dev_B);
        clReleaseMemObject(dev_C);

        clReleaseContext(context);
    }

    free(devices);

    return 0;
}
