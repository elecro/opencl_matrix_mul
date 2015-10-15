#include <stdio.h>
#include <CL/cl.h>


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
    const int matrix_size = sizeof(float) * 10 * 10;
    float* A = allocate_matrix(10, 10);
    float* B = allocate_matrix(10, 10);
    float* C = allocate_matrix(10, 10);

    fill_matrix(A, 10, 10, 2);
    fill_matrix(B, 10, 10, 2);
    fill_matrix(C, 10, 10, 0);

    cl_mem dev_A;
    cl_mem dev_B;
    cl_mem dev_C;

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

        dev_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrix_size, A, &error);
        dev_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matrix_size, B, &error);

        dev_C = clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_size, NULL, &error);

        // TODO: test buffer errors

        int width_A = 10;
        int height_B = 10;

        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&dev_A);
        error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&dev_B);
        error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&dev_C);
        error |= clSetKernelArg(kernel, 3, sizeof(int), (void*)&width_A);
        error |= clSetKernelArg(kernel, 4, sizeof(int), (void*)&height_B);

        if (error != CL_SUCCESS)
        {
            printf("Error: Unable to set kernel arguments\n");
            return -6;
        }

        size_t localWorkSize[2] = { 1, 1 };
        size_t globalWorkSize[2] = { 10, 10 };

        error = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

        if (error != CL_SUCCESS)
        {
            printf("Error: Unable to execute kernel\n");
            return -7;
        }

        error = clEnqueueReadBuffer(queue, dev_C, CL_TRUE, 0, matrix_size, C, 0, NULL, NULL);
        if (error != CL_SUCCESS)
        {
            printf("Error: Unable to read result array\n");
            return -8;
        }

        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                printf("%.2f ", C[i * 10 + j]);
            }
            printf("\n");
        }

        clReleaseMemObject(dev_A);
        clReleaseMemObject(dev_B);
        clReleaseMemObject(dev_C);

        clReleaseContext(context);
    }

    free(devices);

    return 0;
}
