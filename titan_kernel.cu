//
// NOTE: compile this .cu module for compute_35,sm_35 with --maxrregcount=64
//

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <stdio.h>
#ifdef WIN32
#include <windows.h>
#endif
#include <time.h>
#include <sys/time.h>

#include <map>
#include <algorithm>

#include "miner.h"

typedef unsigned int uint32_t;

// Define work unit size
#define WU_PER_WARP 32
#define WU_PER_BLOCK (WU_PER_WARP*WARPS_PER_BLOCK)
#define WU_PER_LAUNCH (GRID_BLOCKS*WU_PER_BLOCK)
#define SCRATCH (32768+16)

// Not performing error checking is actually bad, but...
#define checkCudaErrors(x) x
#define getLastCudaError(x)

// from salsa_kernel.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

__constant__ const uint32_t* __restrict__ c_V[1024];

#define ROTL(a, b) __funnelshift_l( a, a, b );

static __device__ __inline__ void lock(int *mutex, int i)
{
    while( atomicCAS( &mutex[i], 0, 1 ) != 0 );
}

static __device__ __inline__ void unlock(int *mutex, int i)
{
    atomicExch( &mutex[i], 0 );
}

static __device__ void xor_salsa8(uint32_t *B, const uint32_t *C)
{
    uint32_t x0 = (B[ 0] ^= C[ 0]), x1 = (B[ 1] ^= C[ 1]), x2 = (B[ 2] ^= C[ 2]), x3 = (B[ 3] ^= C[ 3]);
    uint32_t x4 = (B[ 4] ^= C[ 4]), x5 = (B[ 5] ^= C[ 5]), x6 = (B[ 6] ^= C[ 6]), x7 = (B[ 7] ^= C[ 7]);
    uint32_t x8 = (B[ 8] ^= C[ 8]), x9 = (B[ 9] ^= C[ 9]), xa = (B[10] ^= C[10]), xb = (B[11] ^= C[11]);
    uint32_t xc = (B[12] ^= C[12]), xd = (B[13] ^= C[13]), xe = (B[14] ^= C[14]), xf = (B[15] ^= C[15]);

    for (int i = 0; i < 4; ++i) {
        /* Operate on columns. */
        x4 ^= ROTL(x0 + xc,  7);  x9 ^= ROTL(x5 + x1,  7); xe ^= ROTL(xa + x6,  7);  x3 ^= ROTL(xf + xb,  7);
        x8 ^= ROTL(x4 + x0,  9);  xd ^= ROTL(x9 + x5,  9); x2 ^= ROTL(xe + xa,  9);  x7 ^= ROTL(x3 + xf,  9);
        xc ^= ROTL(x8 + x4, 13);  x1 ^= ROTL(xd + x9, 13); x6 ^= ROTL(x2 + xe, 13);  xb ^= ROTL(x7 + x3, 13);
        x0 ^= ROTL(xc + x8, 18);  x5 ^= ROTL(x1 + xd, 18); xa ^= ROTL(x6 + x2, 18);  xf ^= ROTL(xb + x7, 18);
        
        /* Operate on rows. */
        x1 ^= ROTL(x0 + x3,  7);  x6 ^= ROTL(x5 + x4,  7); xb ^= ROTL(xa + x9,  7);  xc ^= ROTL(xf + xe,  7);
        x2 ^= ROTL(x1 + x0,  9);  x7 ^= ROTL(x6 + x5,  9); x8 ^= ROTL(xb + xa,  9);  xd ^= ROTL(xc + xf,  9);
        x3 ^= ROTL(x2 + x1, 13);  x4 ^= ROTL(x7 + x6, 13); x9 ^= ROTL(x8 + xb, 13);  xe ^= ROTL(xd + xc, 13);
        x0 ^= ROTL(x3 + x2, 18);  x5 ^= ROTL(x4 + x7, 18); xa ^= ROTL(x9 + x8, 18);  xf ^= ROTL(xe + xd, 18);
    }

    B[ 0] += x0; B[ 1] += x1; B[ 2] += x2; B[ 3] += x3; B[ 4] += x4; B[ 5] += x5; B[ 6] += x6; B[ 7] += x7;
    B[ 8] += x8; B[ 9] += x9; B[10] += xa; B[11] += xb; B[12] += xc; B[13] += xd; B[14] += xe; B[15] += xf;
}


////////////////////////////////////////////////////////////////////////////////
//! Scrypt core kernel with spinlock guards around a smaller shared memory
//! Version for Geforce Titan, low register count (<=64), low shared mem use.
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
scrypt_core_kernel_spinlock_titanA(uint32_t *g_idata, int *mutex)
{
    __shared__ uint32_t X[WU_PER_WARP][16+1]; // +1 to resolve bank conflicts

    int warpIdx         = threadIdx.x / warpSize;
    int warpThread      = threadIdx.x % warpSize;
    int halfwarpOffset  = warpThread/16;
    int halfwarpThread  = warpThread%16;
    int WARPS_PER_BLOCK = blockDim.x / warpSize;

    // add block specific offsets
    int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_idata += 32 * offset + halfwarpThread;
    uint32_t *V = (uint32_t *) c_V[offset/WU_PER_WARP] + halfwarpThread;

    // registers to store an entire work unit
    uint32_t B[16], C[16];

    if (warpThread == 0) lock(mutex, blockIdx.x);
#pragma unroll 16
    for (int wu=0; wu < 16; wu++)
        V[SCRATCH*(2*wu+halfwarpOffset) + 0*32] = X[2*wu+halfwarpOffset][halfwarpThread] = g_idata[32*(2*wu+halfwarpOffset)];
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) B[idx] = X[warpThread][idx];

#pragma unroll 16
    for (int wu=0; wu < 16; wu++)
        V[SCRATCH*(2*wu+halfwarpOffset) + 0*32 + 16] = X[2*wu+halfwarpOffset][halfwarpThread] = g_idata[32*(2*wu+halfwarpOffset)+16];
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) C[idx] = X[warpThread][idx];

    for (int i = 1; i < 1024; i++) {

        if (warpThread == 0) unlock(mutex, blockIdx.x);
        xor_salsa8(B, C); xor_salsa8(C, B);
        if (warpThread == 0) lock(mutex, blockIdx.x);

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpThread][idx] = B[idx];
#pragma unroll 1
        for (int wu=0; wu < 16; wu++)
            V[SCRATCH*(2*wu+halfwarpOffset) + i*32] = X[2*wu+halfwarpOffset][halfwarpThread];

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpThread][idx] = C[idx];
#pragma unroll 1
        for (int wu=0; wu < 16; wu++)
            V[SCRATCH*(2*wu+halfwarpOffset) + i*32 + 16] = X[2*wu+halfwarpOffset][halfwarpThread];
    }
    if (warpThread == 0) unlock(mutex, blockIdx.x);
}

__global__ void
scrypt_core_kernel_spinlock_titanB(uint32_t *g_odata, int *mutex)
{
    __shared__ uint32_t X[WU_PER_WARP][16+1]; // +1 to resolve bank conflicts

    int warpIdx         = threadIdx.x / warpSize;
    int warpThread      = threadIdx.x % warpSize;
    int halfwarpOffset  = warpThread/16;
    int halfwarpThread  = warpThread%16;
    int WARPS_PER_BLOCK = blockDim.x / warpSize;

    // add block specific offsets
    int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_odata += 32 * offset + halfwarpThread;
    const uint32_t* __restrict__ V = c_V[offset/WU_PER_WARP] + halfwarpThread;

    // registers to store an entire work unit
    uint32_t B[16], C[16];

    if (warpThread == 0) lock(mutex, blockIdx.x);
#pragma unroll 16
    for (int wu=0; wu < 16; wu++)
        X[2*wu+halfwarpOffset][halfwarpThread] = V[SCRATCH*(2*wu+halfwarpOffset) + 1023*32];
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) B[idx] = X[warpThread][idx];

#pragma unroll 16
    for (int wu=0; wu < 16; wu++)
        X[2*wu+halfwarpOffset][halfwarpThread] = V[SCRATCH*(2*wu+halfwarpOffset) + 1023*32 + 16];
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) C[idx] = X[warpThread][idx];

    if (warpThread == 0) unlock(mutex, blockIdx.x);
    xor_salsa8(B, C); xor_salsa8(C, B);
    if (warpThread == 0) lock(mutex, blockIdx.x);

    for (int i = 0; i < 1024; i++) {

        X[warpThread][16] = C[0];

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpThread][idx] = B[idx];
#pragma unroll 1
        for (int wu=0; wu < 16; wu++)
            X[2*wu+halfwarpOffset][halfwarpThread] ^= V[SCRATCH*(2*wu+halfwarpOffset) + 32*(X[2*wu+halfwarpOffset][16] & 1023)];
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) B[idx] = X[warpThread][idx];

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpThread][idx] = C[idx];
#pragma unroll 1
        for (int wu=0; wu < 16; wu++)
            X[2*wu+halfwarpOffset][halfwarpThread] ^= V[SCRATCH*(2*wu+halfwarpOffset) + 32*(X[2*wu+halfwarpOffset][16] & 1023) + 16];
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) C[idx] = X[warpThread][idx];

        if (warpThread == 0) unlock(mutex, blockIdx.x);
        xor_salsa8(B, C); xor_salsa8(C, B);
        if (warpThread == 0) lock(mutex, blockIdx.x);
    }

#pragma unroll 16
    for (int idx=0; idx < 16; ++idx) X[warpThread][idx] = B[idx];
#pragma unroll 16
    for (int wu=0; wu < 16; wu++)
        g_odata[32*(2*wu+halfwarpOffset)] = X[2*wu+halfwarpOffset][halfwarpThread];

#pragma unroll 16
    for (int idx=0; idx < 16; ++idx) X[warpThread][idx] = C[idx];
#pragma unroll 16
    for (int wu=0; wu < 16; wu++)
        g_odata[32*(2*wu+halfwarpOffset)+16] = X[2*wu+halfwarpOffset][halfwarpThread];

    if (warpThread == 0) unlock(mutex, blockIdx.x);
}

////////////////////////////////////////////////////////////////////////////////
//! Scrypt core kernel
//! Version for Geforce Titan, low register count (<=64), low shared mem use.
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
template <int WARPS_PER_BLOCK> __global__ void
scrypt_core_kernel_titanA(uint32_t *g_idata)
{
    __shared__ uint32_t X[WARPS_PER_BLOCK][WU_PER_WARP][16+1]; // +1 to resolve bank conflicts

    int warpIdx        = threadIdx.x / warpSize;
    int warpThread     = threadIdx.x % warpSize;
    int halfwarpOffset = warpThread/16;
    int halfwarpThread = warpThread%16;

    // add block specific offsets
    int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_idata += 32 * offset + halfwarpThread;
    uint32_t *V = (uint32_t *) c_V[offset/WU_PER_WARP] + halfwarpThread;

    // registers to store an entire work unit
    uint32_t B[16], C[16];

#pragma unroll 16
    for (int wu=0; wu < 16; wu++)
        V[SCRATCH*(2*wu+halfwarpOffset) + 0*32] = X[warpIdx][2*wu+halfwarpOffset][halfwarpThread] = g_idata[32*(2*wu+halfwarpOffset)];
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) B[idx] = X[warpIdx][warpThread][idx];

#pragma unroll 16
    for (int wu=0; wu < 16; wu++)
        V[SCRATCH*(2*wu+halfwarpOffset) + 0*32 + 16] = X[warpIdx][2*wu+halfwarpOffset][halfwarpThread] = g_idata[32*(2*wu+halfwarpOffset)+16];
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) C[idx] = X[warpIdx][warpThread][idx];

    for (int i = 1; i < 1024; i++) {

        xor_salsa8(B, C); xor_salsa8(C, B);

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpIdx][warpThread][idx] = B[idx];
#pragma unroll 1
        for (int wu=0; wu < 16; wu++)
            V[SCRATCH*(2*wu+halfwarpOffset) + i*32] = X[warpIdx][2*wu+halfwarpOffset][halfwarpThread];

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpIdx][warpThread][idx] = C[idx];
#pragma unroll 1
        for (int wu=0; wu < 16; wu++)
            V[SCRATCH*(2*wu+halfwarpOffset) + i*32 + 16] = X[warpIdx][2*wu+halfwarpOffset][halfwarpThread];
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Scrypt core kernel
//! Version for Geforce Titan, low register count (<=64), low shared mem use.
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
template <int WARPS_PER_BLOCK> __global__ void
scrypt_core_kernel_titanB(uint32_t *g_odata)
{
    __shared__ uint32_t X[WARPS_PER_BLOCK][WU_PER_WARP][16+1]; // +1 to resolve bank conflicts

    int warpIdx        = threadIdx.x / warpSize;
    int warpThread     = threadIdx.x % warpSize;
    int halfwarpOffset = warpThread/16;
    int halfwarpThread = warpThread%16;

    // add block specific offsets
    int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_odata += 32 * offset + halfwarpThread;
    const uint32_t* __restrict__ V = c_V[offset/WU_PER_WARP] + halfwarpThread;

    // registers to store an entire work unit
    uint32_t B[16], C[16];

#pragma unroll 16
    for (int wu=0; wu < 16; wu++)
        X[warpIdx][2*wu+halfwarpOffset][halfwarpThread] = V[SCRATCH*(2*wu+halfwarpOffset) + 1023*32];
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) B[idx] = X[warpIdx][warpThread][idx];

#pragma unroll 16
    for (int wu=0; wu < 16; wu++)
        X[warpIdx][2*wu+halfwarpOffset][halfwarpThread] = V[SCRATCH*(2*wu+halfwarpOffset) + 1023*32 + 16];
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) C[idx] = X[warpIdx][warpThread][idx];

    xor_salsa8(B, C); xor_salsa8(C, B);

    for (int i = 0; i < 1024; i++) {

        X[warpIdx][warpThread][16] = C[0];

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpIdx][warpThread][idx] = B[idx];
#pragma unroll 1
        for (int wu=0; wu < 16; wu++)
            X[warpIdx][2*wu+halfwarpOffset][halfwarpThread] ^= V[SCRATCH*(2*wu+halfwarpOffset) + 32*(X[warpIdx][2*wu+halfwarpOffset][16] & 1023)];
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) B[idx] = X[warpIdx][warpThread][idx];

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpIdx][warpThread][idx] = C[idx];
#pragma unroll 1
        for (int wu=0; wu < 16; wu++)
            X[warpIdx][2*wu+halfwarpOffset][halfwarpThread] ^= V[SCRATCH*(2*wu+halfwarpOffset) + 32*(X[warpIdx][2*wu+halfwarpOffset][16] & 1023) + 16];
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) C[idx] = X[warpIdx][warpThread][idx];

        xor_salsa8(B, C); xor_salsa8(C, B);
    }

#pragma unroll 16
    for (int idx=0; idx < 16; ++idx) X[warpIdx][warpThread][idx] = B[idx];
#pragma unroll 16
    for (int wu=0; wu < 16; wu++)
        g_odata[32*(2*wu+halfwarpOffset)] = X[warpIdx][2*wu+halfwarpOffset][halfwarpThread];

#pragma unroll 16
    for (int idx=0; idx < 16; ++idx) X[warpIdx][warpThread][idx] = C[idx];
#pragma unroll 16
    for (int wu=0; wu < 16; wu++)
        g_odata[32*(2*wu+halfwarpOffset)+16] = X[warpIdx][2*wu+halfwarpOffset][halfwarpThread];
}


void set_titan_scratchbuf_constants(int MAXWARPS, uint32_t** h_V)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_V, h_V, MAXWARPS*sizeof(uint32_t*), 0, cudaMemcpyHostToDevice));
}

bool run_titan_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, int *mutex, bool special, bool interactive, bool benchmark)
{
    bool success = true;

    // clear CUDA's error variable
    cudaGetLastError();

    // First phase: Sequential writes to scratchpad.

    if (special)
        scrypt_core_kernel_spinlock_titanA<<< grid, threads, 0, stream >>>(d_idata, mutex);
    else
        switch (WARPS_PER_BLOCK) {
            case 1: scrypt_core_kernel_titanA<1><<< grid, threads, 0, stream >>>(d_idata); break;
            case 2: scrypt_core_kernel_titanA<2><<< grid, threads, 0, stream >>>(d_idata); break;
            case 3: scrypt_core_kernel_titanA<3><<< grid, threads, 0, stream >>>(d_idata); break;
            case 4: scrypt_core_kernel_titanA<4><<< grid, threads, 0, stream >>>(d_idata); break;
            case 5: scrypt_core_kernel_titanA<5><<< grid, threads, 0, stream >>>(d_idata); break;
            case 6: scrypt_core_kernel_titanA<6><<< grid, threads, 0, stream >>>(d_idata); break;
            case 7: scrypt_core_kernel_titanA<7><<< grid, threads, 0, stream >>>(d_idata); break;
            case 8: scrypt_core_kernel_titanA<8><<< grid, threads, 0, stream >>>(d_idata); break;
            default: success = false; break;
        }

    // Optional millisecond sleep in between kernels

    if (!benchmark && interactive) {
        checkCudaErrors(MyStreamSynchronize(stream, 1, thr_id));
#ifdef WIN32
        Sleep(1);
#else
        usleep(1000);
#endif
    }

    // Second phase: Random read access from scratchpad.

    if (special)
        scrypt_core_kernel_spinlock_titanB<<< grid, threads, 0, stream >>>(d_odata, mutex);
    else
        switch (WARPS_PER_BLOCK) {
            case 1: scrypt_core_kernel_titanB<1><<< grid, threads, 0, stream >>>(d_odata); break;
            case 2: scrypt_core_kernel_titanB<2><<< grid, threads, 0, stream >>>(d_odata); break;
            case 3: scrypt_core_kernel_titanB<3><<< grid, threads, 0, stream >>>(d_odata); break;
            case 4: scrypt_core_kernel_titanB<4><<< grid, threads, 0, stream >>>(d_odata); break;
            case 5: scrypt_core_kernel_titanB<5><<< grid, threads, 0, stream >>>(d_odata); break;
            case 6: scrypt_core_kernel_titanB<6><<< grid, threads, 0, stream >>>(d_odata); break;
            case 7: scrypt_core_kernel_titanB<7><<< grid, threads, 0, stream >>>(d_odata); break;
            case 8: scrypt_core_kernel_titanB<8><<< grid, threads, 0, stream >>>(d_odata); break;
            default: success = false; break;
        }

    // catch any kernel launch failures
    if (cudaPeekAtLastError() != cudaSuccess) success = false;

    return success;
}
