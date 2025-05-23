#ifndef CUDA_TOOLS_HPP
#define CUDA_TOOLS_HPP

/*
 *  系统关于CUDA的功能函数
 */

#include "nodes/stream/ffhdd/simple-logger.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

#define GPU_BLOCK_THREADS 512

#define KernelPositionBlock                                                                                            \
    int position = (blockDim.x * blockIdx.x + threadIdx.x);                                                            \
    if (position >= (edge))                                                                                            \
        return;

#define checkCudaDriver(call) CUDATools::check_driver(call, #call, __LINE__, __FILE__)
#define checkCudaRuntime(call) CUDATools::check_runtime(call, #call, __LINE__, __FILE__)

#define checkCudaKernel(...)                                                                                           \
    __VA_ARGS__;                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = cudaPeekAtLastError();                                                                \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            INFOE("launch failed: %s", cudaGetErrorString(cudaStatus));                                                \
        }                                                                                                              \
    } while (0);

#define Assert(op)                                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        bool cond = !(!(op));                                                                                          \
        if (!cond)                                                                                                     \
        {                                                                                                              \
            INFOF("Assert failed, " #op);                                                                              \
        }                                                                                                              \
    } while (false)

struct CUctx_st;
struct CUstream_st;

typedef CUstream_st *ICUStream;
typedef CUctx_st *ICUContext;
typedef void *ICUDeviceptr;
typedef int DeviceID;

namespace CUDATools
{

bool check_driver(CUresult e, const char *call, int iLine, const char *szFile);
bool check_runtime(cudaError_t e, const char *call, int iLine, const char *szFile);
bool check_device_id(int device_id);
int current_device_id();

dim3 grid_dims(int numJobs);
dim3 block_dims(int numJobs);

class AutoDevice
{
  public:
    AutoDevice(int device_id = 0);
    virtual ~AutoDevice();

  private:
    int old_ = -1;
};
} // namespace CUDATools

#endif // CUDA_TOOLS_HPP