#ifndef _OPERATIONS_HPP
#define _OPERATIONS_HPP

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include "matrix.hpp"
#include "mem.hpp"

class Operations
{
public:
    virtual ~Operations(void) { }
    virtual Matrix multiply(const Matrix& lhs, const Matrix& rhs) const = 0;
};

class CpuOperations : public Operations
{
public:
    virtual Matrix multiply(const Matrix& lhs, const Matrix& rhs) const;
};

class GpuOperations : public Operations
{
private:
    cl_device_id m_deviceId;

public:
    GpuOperations(void);

    virtual Matrix multiply(const Matrix& lhs, const Matrix& rhs) const;

protected:
    GpuOperations(std::string kernelFile);
    cl_context createContext(void) const;
    cl_command_queue createCommandQueue(cl_context context) const;
    cl_program buildProgram(cl_context context, const std::string& filename) const;
    cl_kernel createKernel(cl_context context, cl_program program, const std::string& name) const;
    cl_mem uploadBuffer(cl_context context, size_t size, const void* dataPtr) const;

    static cl_device_id selectDevice(void);

    CleanUp<cl_context> m_context;
    CleanUp<cl_command_queue> m_queue;
    CleanUp<cl_program> m_program;
};

class TransposedGpuOperations : public GpuOperations
{
public:
    TransposedGpuOperations(void);

    virtual Matrix multiply(const Matrix& lhs, const Matrix& rhs) const;
};

class DotGpuOperations : public GpuOperations
{
public:
    DotGpuOperations(void)
        : GpuOperations("matrix_mul_dot.cl")
    {}

};

class Float4GpuOperations : public GpuOperations
{
public:
    Float4GpuOperations(void)
        : GpuOperations("matrix_mul_float4.cl")
    {}
};

class ConstantGpuOperations : public GpuOperations
{
public:
    ConstantGpuOperations(void)
        : GpuOperations("matrix_mul_constant.cl")
    {}
};

#endif // _OPERATIONS_HPP
