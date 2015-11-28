#include "operations.hpp"

static char* query_device_info(cl_device_id device, cl_device_info value_param)
{
    size_t value_size = 0;
    clGetDeviceInfo(device, value_param, 0, NULL, &value_size);
    char* value = (char*)malloc(sizeof(char) * value_size);
    clGetDeviceInfo(device, value_param, value_size, value, NULL);

    return value;
}

static int device_is_gpu(cl_device_id device)
{
    cl_device_type device_type = CL_DEVICE_TYPE_DEFAULT;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);

    return device_type == CL_DEVICE_TYPE_GPU ? 1 : 0;
}

char* read_file(const char* filename)
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

// Operations

// CPU

Matrix CpuOperations::multiply(const Matrix& lhs, const Matrix& rhs) const
{
    int lhsWidth = lhs.width();
    int rhsWidth = rhs.width();

    int width = rhs.width();
    int height = lhs.height();
    std::vector<float> data(width * height);
    for (int i = 0; i < width * height; i++)
    {
        int x = i % width;
        int y = i / width;
        float value = 0;

        for (int j = 0; j < lhsWidth; j++)
        {
            value += lhs[y * lhsWidth + j] * rhs[j * rhsWidth + x];
        }

        data[i] = value;
    }

    return Matrix(width, height, data);
}

// GPU

GpuOperations::GpuOperations(void)
    : m_deviceId(selectDevice())
    , m_context(createContext())
    , m_queue(createCommandQueue(m_context.get()))
    , m_program(buildProgram(m_context.get(), "matrix_mul.cl"))
{
}

GpuOperations::GpuOperations(std::string kernelFile)
    : m_deviceId(selectDevice())
    , m_context(createContext())
    , m_queue(createCommandQueue(m_context.get()))
    , m_program(buildProgram(m_context.get(), kernelFile))
{
}

Matrix GpuOperations::multiply(const Matrix& lhs, const Matrix& rhs) const
{
    const int width = rhs.width();
    const int height = lhs.height();
    const int dataSize = sizeof(float) * width * height;

    // Prepare kernel
    CleanUp<cl_kernel> kernel = createKernel(m_context.get(), m_program.get(), "matrix_mul");

    // Prepare kernel arguments
    CleanUp<cl_mem> dev_A = uploadBuffer(m_context.get(), lhs.dataSize(), lhs.data());
    CleanUp<cl_mem> dev_B = uploadBuffer(m_context.get(), rhs.dataSize(), rhs.data());
    CleanUp<cl_mem> dev_C = uploadBuffer(m_context.get(), dataSize, NULL);

    const cl_mem memObjs[] = { dev_A.get(), dev_B.get(), dev_C.get(), 0 };
    int argNdx = 0;
    for (int i = 0; memObjs[i] != 0; i++, argNdx++)
    {
        if (clSetKernelArg(kernel.get(), argNdx, sizeof(cl_mem), (void*)&memObjs[i]) != CL_SUCCESS)
        {
            throw "kernel arg set fail";
        }
    }

    const int sizes[] = { lhs.width(), rhs.width(), 0 };
    for (int i = 0; sizes[i] != 0; i++, argNdx++)
    {
        if (clSetKernelArg(kernel.get(), argNdx, sizeof(int), (void*)&sizes[i]) != CL_SUCCESS)
        {
            throw "kernel arg set fail";
        }
    }

    size_t globalWorkSize[2] = { (size_t)width, (size_t)height };

    if (clEnqueueNDRangeKernel(m_queue.get(), kernel.get(), 2, NULL, globalWorkSize, NULL /* localWorkSize */, 0, NULL, NULL) != CL_SUCCESS)
    {
        throw "job enqueue fail";
    }

    std::vector<float> data(width * height);

    if (clEnqueueReadBuffer(m_queue.get(), dev_C.get(), CL_TRUE, 0, dataSize, &data[0], 0, NULL, NULL) != CL_SUCCESS)
    {
        throw "readback fail";
    }

    return Matrix(width, height, data);
}

cl_context GpuOperations::createContext(void) const
{
    cl_int error;
    cl_context context = clCreateContext(NULL, 1, &m_deviceId, NULL, NULL, &error);
    if (!context || error != CL_SUCCESS)
    {
        throw "context fail";
    }

    return context;
}

cl_command_queue GpuOperations::createCommandQueue(cl_context context) const
{
    cl_int error;
    cl_command_queue queue = clCreateCommandQueue(context, m_deviceId, 0, &error);
    if (!queue || error != CL_SUCCESS)
    {
        throw "command queue fail";
    }

    return queue;
}

cl_program GpuOperations::buildProgram(cl_context context, const std::string& filename) const
{
    cl_int error;
    char* kernel_source = read_file(filename.c_str());
    const char* src = kernel_source;
    cl_program program = clCreateProgramWithSource(context, 1, &src, NULL, &error);
    free(kernel_source);
    if (!program || error != CL_SUCCESS)
    {
        throw "program fail";
    }

    error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (error)
    {
        size_t len;
        char buffer[2048];
        printf("Error: Unable to build the program, error message:\n");
        clGetProgramBuildInfo(program, m_deviceId, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        throw "build fail";
    }

    return program;
}

cl_kernel GpuOperations::createKernel(cl_context context, cl_program program, const std::string& name) const
{
    cl_int error;
    cl_kernel kernel = clCreateKernel(program, name.c_str(), &error);
    if (!kernel || error != CL_SUCCESS)
    {
        throw "kernel fail";
    }

    return kernel;
}

cl_mem GpuOperations::uploadBuffer(cl_context context, size_t size, const void* dataPtr) const
{
    cl_int error;
    cl_mem_flags flags = CL_MEM_READ_WRITE | (dataPtr != NULL ? CL_MEM_COPY_HOST_PTR : 0);
    cl_mem mem = clCreateBuffer(context, flags, size, const_cast<void*>(dataPtr), &error);
    if (error != CL_SUCCESS)
    {
        throw "upload buffer fail";
    }
    return mem;
}

cl_device_id GpuOperations::selectDevice(void)
{
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
        /*
        printf("%d. Device name: %s (HW: %s, SW: %s, Is gpu?: %s)\n",
                j + 1,
                name,
                hw_opencl,
                sw_opencl,
                is_gpu ? "Yes" : "No");
        */
        if (is_gpu && selected_device == NULL)
        {
            selected_device = devices[j];
            /* printf(" Selected device ^^\n"); */
        }

        free(name);
        free(hw_opencl);
        free(sw_opencl);
    }
    free(devices);

    return selected_device;
}


// Transposed Gpu Operations

TransposedGpuOperations::TransposedGpuOperations(void)
    : GpuOperations("matrix_mul_transposed.cl")
{
}

Matrix TransposedGpuOperations::multiply(const Matrix& lhs, const Matrix& rhs) const
{
    const int width = rhs.width();
    const int height = lhs.height();
    const int dataSize = sizeof(float) * width * height;

    // Prepare kernel
    CleanUp<cl_kernel> kernel = createKernel(m_context.get(), m_program.get(), "matrix_mul");
    CleanUp<cl_kernel> transposeKernel = createKernel(m_context.get(), m_program.get(), "matrix_transpose");

    // Prepare kernel arguments
    CleanUp<cl_mem> dev_A = uploadBuffer(m_context.get(), lhs.dataSize(), lhs.data());
    CleanUp<cl_mem> dev_B = uploadBuffer(m_context.get(), rhs.dataSize(), rhs.data());
    // Transposed dst matrix
    CleanUp<cl_mem> dev_T = uploadBuffer(m_context.get(), rhs.dataSize(), NULL);
    CleanUp<cl_mem> dev_C = uploadBuffer(m_context.get(), dataSize, NULL);

    {
        const cl_mem memObjs[] = { dev_B.get(), dev_T.get(), 0 };
        int argNdx = 0;
        for (int i = 0; memObjs[i] != 0; i++, argNdx++)
        {
            if (clSetKernelArg(transposeKernel.get(), argNdx, sizeof(cl_mem), (void*)&memObjs[i]) != CL_SUCCESS)
            {
                throw "kernel arg set fail";
            }
        }
    }

    size_t transposeWorkSize[2] = { (size_t)rhs.width(), (size_t)rhs.height() };
    cl_event transposeEvent;
    cl_int error = clEnqueueNDRangeKernel(m_queue.get(), transposeKernel.get(), 2, NULL, transposeWorkSize, NULL, 0, NULL, &transposeEvent);
    if (error != CL_SUCCESS)
    {
        throw "transpose job enqueue fail";
    }

    {
        const cl_mem memObjs[] = { dev_A.get(), dev_T.get(), dev_C.get(), 0 };
        int argNdx = 0;
        for (int i = 0; memObjs[i] != 0; i++, argNdx++)
        {
            if (clSetKernelArg(kernel.get(), argNdx, sizeof(cl_mem), (void*)&memObjs[i]) != CL_SUCCESS)
            {
                throw "kernel arg set fail";
            }
        }

        const int sizes[] = { lhs.width(), rhs.width(), 0 };
        for (int i = 0; sizes[i] != 0; i++, argNdx++)
        {
            if (clSetKernelArg(kernel.get(), argNdx, sizeof(int), (void*)&sizes[i]) != CL_SUCCESS)
            {
                throw "kernel arg set fail";
            }
        }
    }

    size_t globalWorkSize[2] = { (size_t)width, (size_t)height };
    if (clEnqueueNDRangeKernel(m_queue.get(), kernel.get(), 2, NULL, globalWorkSize, NULL, 1, &transposeEvent, NULL) != CL_SUCCESS)
    {
        throw "job enqueue fail";
    }

    std::vector<float> data(width * height);

    if (clEnqueueReadBuffer(m_queue.get(), dev_C.get(), CL_TRUE, 0, dataSize, &data[0], 0, NULL, NULL) != CL_SUCCESS)
    {
        throw "readback fail";
    }

    return Matrix(width, height, data);
}

// Dot GPU Operations
DotGpuOperations::DotGpuOperations (void)
    : GpuOperations("matrix_mul_dot.cl")
{
}
