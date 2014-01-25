#ifndef SALSA_KERNEL_H
#define SALSA_KERNEL_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned int uint32_t; // define this as 32 bit type derived from int

// from cpu-miner.h
extern int device_map[8];
extern int device_interactive[8];
extern int device_batchsize[8];
extern int device_backoff[8];
extern int device_lookup_gap[8];
extern int device_texturecache[8];
extern int device_singlememory[8];
extern char *device_config[8];
extern char *device_name[8];
extern bool autotune;

// CUDA externals
extern int cuda_num_devices();
extern int cuda_finddevice(char *name);
extern void cuda_shutdown(int thr_id);
extern int cuda_throughput(int thr_id);

extern uint32_t *cuda_transferbuffer(int thr_id, int stream);
extern uint32_t *cuda_hashbuffer(int thr_id, int stream);

extern void cuda_scrypt_HtoD(int thr_id, uint32_t *X, int stream);
extern void cuda_scrypt_serialize(int thr_id, int stream);
extern void cuda_scrypt_core(int thr_id, int stream, unsigned int N);
extern void cuda_scrypt_done(int thr_id, int stream);
extern void cuda_scrypt_DtoH(int thr_id, uint32_t *X, int stream);
extern void cuda_scrypt_sync(int thr_id, int stream);
extern void cuda_scrypt_flush(int thr_id, int stream);

extern void computeGold(uint32_t *idata, uint32_t *reference, uint32_t *V);

#ifdef __NVCC__
#include <cuda_runtime.h>
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);
#endif

#ifdef __cplusplus
}

// If we're in C++ mode, we're either compiling .cu files or scrypt.cpp

#ifdef __NVCC__
/**
 * An pure virtual interface for a CUDA kernel implementation.
 * TODO: encapsulate the kernel launch parameters in some kind of wrapper.
 */
class KernelInterface
{
public:
    virtual void set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V) = 0;
    virtual bool run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, unsigned int N, unsigned int LOOKUP_GAP, bool interactive, bool benchmark, int texture_cache) = 0;
    virtual bool bindtexture_1D(uint32_t *d_V, size_t size) { return true; }
    virtual bool bindtexture_2D(uint32_t *d_V, int width, int height, size_t pitch) { return true; }
    virtual bool unbindtexture_1D() { return true; }
    virtual bool unbindtexture_2D() { return true; }

    virtual char get_identifier() = 0;
    virtual int get_major_version() { return 1; }
    virtual int get_minor_version() { return 0; }
    virtual int max_warps_per_block() = 0;
    virtual int get_texel_width() = 0;
    virtual bool no_textures() { return false; };
    virtual bool single_memory() { return false; };
    virtual int threads_per_wu() { return 1; }
    virtual bool support_lookup_gap() { return false; }
    virtual cudaSharedMemConfig shared_mem_config() { return cudaSharedMemBankSizeDefault; }
    virtual cudaFuncCache cache_config() { return cudaFuncCachePreferNone; }
};

// Define work unit size
#define TOTAL_WARP_LIMIT 4096
#define WU_PER_WARP (32 / THREADS_PER_WU)
#define WU_PER_BLOCK (WU_PER_WARP*WARPS_PER_BLOCK)
#define WU_PER_LAUNCH (GRID_BLOCKS*WU_PER_BLOCK)

// make scratchpad size dependent on N and LOOKUP_GAP
#define SCRATCH (((N+LOOKUP_GAP-1)/LOOKUP_GAP)*32)

// Not performing error checking is actually bad, but...
#define checkCudaErrors(x) x
#define getLastCudaError(x)

#endif // #ifdef __NVCC__
#endif // #ifdef __cplusplus

#endif // #ifndef SALSA_KERNEL_H
