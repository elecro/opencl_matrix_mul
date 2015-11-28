#ifndef _MATRIX_HPP
#define _MATRIX_HPP

#include <vector>
#include <string>

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include "mem.hpp"

class Matrix
{
public:
    Matrix(int width, int height);
    Matrix(int width, int height, float filler);
    Matrix(const Matrix& matrix);
    Matrix(int width, int height, std::vector<float> data);


    static Matrix random(int width, int height, int limit);

    virtual int width(void) const { return m_width; }
    virtual int height(void) const { return m_height; }

    virtual float operator[](int idx) const { return m_data[idx]; }
    virtual const float *data(void) const { return &m_data[0]; }
    virtual int dataSize(void) const { return sizeof(float) * m_width * m_height; }

    Matrix transpose(void) const;

    bool operator==(const Matrix& other);
    bool operator!=(const Matrix& other) { return !this->operator==(other); }

private:
    Matrix& operator=(Matrix&);

    int m_width;
    int m_height;
    std::vector<float> m_data;
};

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

void print(Matrix& matrix);

#endif // _MATRIX_HPP
