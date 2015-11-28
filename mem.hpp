#ifndef _MEM_HPP
#define _MEM_HPP

#include <cstdio>

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

// CL object release template method
template<typename T>
void Release(T data);

// Object cleanup smart class
template<typename T>
class CleanUp
{
public:
    CleanUp(T data)
        : m_data(data)
    {}

    T get(void) const { return m_data; }

    virtual ~CleanUp(void)
    {
        Release<T>(m_data);
    }

protected:
    T m_data;
};

// CL object release specialized template prototypes

#define RELEASE(T, name) template<> void Release(T data)

RELEASE(cl_context, Context);
RELEASE(cl_command_queue, CommandQueue);
RELEASE(cl_program, Program);
RELEASE(cl_kernel, Kernel);
RELEASE(cl_mem, MemObject);

#undef RELEASE

#endif // _MEM_HPP
