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
public:
    GpuOperations(void);

    virtual Matrix multiply(const Matrix& lhs, const Matrix& rhs) const;

protected:
    cl_context createContext(void) const;
    cl_command_queue createCommandQueue(cl_context context) const;
    cl_program buildProgram(cl_context context, const std::string& filename) const;
    cl_kernel createKernel(cl_context context, cl_program program, const std::string& name) const;
    cl_mem uploadBuffer(cl_context context, size_t size, const void* dataPtr) const;

    static cl_device_id selectDevice(void);

private:
    cl_device_id m_deviceId;
    CleanUp<cl_context> m_context;
    CleanUp<cl_command_queue> m_queue;
};

#endif // _OPERATIONS_HPP
