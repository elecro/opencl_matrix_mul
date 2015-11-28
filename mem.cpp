#include "mem.hpp"

// CL object release specialized templates

#define RELEASE(T, name) \
template<>                          \
void Release(T data)                \
{                                   \
    clRelease ## name (data);       \
}

RELEASE(cl_context, Context)
RELEASE(cl_command_queue, CommandQueue)
RELEASE(cl_program, Program)
RELEASE(cl_kernel, Kernel)
RELEASE(cl_mem, MemObject)

#undef RELEASE
