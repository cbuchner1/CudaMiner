//
// Kernel that runs best on Kepler (Compute 3.5) devices
// uses funnel shifter and __ldg() intrinsic, but suffers from unfavorable
// shared memory alignment (+4 instead of +1) due to different PTX ISA
//
// NOTE: compile this .cu module for compute_35,sm_35 with --maxrregcount=64
//

#ifdef WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include <cuda.h>

#include "titan_kernel.h"

// forward references
template <int WARPS_PER_BLOCK> __global__ void scrypt_core_kernel_titanA(uint32_t *g_idata, int *mutex);
template <int WARPS_PER_BLOCK> __global__ void scrypt_core_kernel_titanB(uint32_t *g_odata, int *mutex);

// scratchbuf constants (pointers to scratch buffer for each work unit)
__constant__ uint32_t* c_V[1024];

TitanKernel::TitanKernel() : KernelInterface()
{
    cudaDeviceSetSharedMemConfig ( cudaSharedMemBankSizeEightByte );
}

void TitanKernel::set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_V, h_V, MAXWARPS*sizeof(uint32_t*), 0, cudaMemcpyHostToDevice));
}

bool TitanKernel::run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, int *mutex, bool interactive, bool benchmark, int texture_cache)
{
    bool success = true;

    // clear CUDA's error variable
    cudaGetLastError();

    // First phase: Sequential writes to scratchpad.

    switch (WARPS_PER_BLOCK) {
        case 1: scrypt_core_kernel_titanA<1><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 2: scrypt_core_kernel_titanA<2><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 3: scrypt_core_kernel_titanA<3><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 4: scrypt_core_kernel_titanA<4><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 5: scrypt_core_kernel_titanA<5><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 6: scrypt_core_kernel_titanA<6><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 7: scrypt_core_kernel_titanA<7><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 8: scrypt_core_kernel_titanA<8><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 9: scrypt_core_kernel_titanA<9><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 10: scrypt_core_kernel_titanA<10><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 11: scrypt_core_kernel_titanA<11><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 12: scrypt_core_kernel_titanA<12><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 13: scrypt_core_kernel_titanA<13><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 14: scrypt_core_kernel_titanA<14><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 15: scrypt_core_kernel_titanA<15><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 16: scrypt_core_kernel_titanA<16><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
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

    switch (WARPS_PER_BLOCK) {
        case 1: scrypt_core_kernel_titanB<1><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 2: scrypt_core_kernel_titanB<2><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 3: scrypt_core_kernel_titanB<3><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 4: scrypt_core_kernel_titanB<4><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 5: scrypt_core_kernel_titanB<5><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 6: scrypt_core_kernel_titanB<6><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 7: scrypt_core_kernel_titanB<7><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 8: scrypt_core_kernel_titanB<8><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 9: scrypt_core_kernel_titanB<9><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 10: scrypt_core_kernel_titanB<10><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 11: scrypt_core_kernel_titanB<11><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 12: scrypt_core_kernel_titanB<12><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 13: scrypt_core_kernel_titanB<13><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 14: scrypt_core_kernel_titanB<14><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 15: scrypt_core_kernel_titanB<15><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 16: scrypt_core_kernel_titanB<16><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        default: success = false; break;
    }

    // catch any kernel launch failures
    if (cudaPeekAtLastError() != cudaSuccess) success = false;

    return success;
}

#define ROTL(a, b) __funnelshift_l( a, a, b );

static __device__  void xor_salsa8(uint32_t* __restrict__ B, const uint32_t* __restrict__ C)
{
    uint32_t x0 = (B[ 0] ^= C[ 0]), x1 = (B[ 1] ^= C[ 1]), x2 = (B[ 2] ^= C[ 2]), x3 = (B[ 3] ^= C[ 3]);
    uint32_t x4 = (B[ 4] ^= C[ 4]), x5 = (B[ 5] ^= C[ 5]), x6 = (B[ 6] ^= C[ 6]), x7 = (B[ 7] ^= C[ 7]);
    uint32_t x8 = (B[ 8] ^= C[ 8]), x9 = (B[ 9] ^= C[ 9]), xa = (B[10] ^= C[10]), xb = (B[11] ^= C[11]);
    uint32_t xc = (B[12] ^= C[12]), xd = (B[13] ^= C[13]), xe = (B[14] ^= C[14]), xf = (B[15] ^= C[15]);

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

    B[ 0] += x0; B[ 1] += x1; B[ 2] += x2; B[ 3] += x3; B[ 4] += x4; B[ 5] += x5; B[ 6] += x6; B[ 7] += x7;
    B[ 8] += x8; B[ 9] += x9; B[10] += xa; B[11] += xb; B[12] += xc; B[13] += xd; B[14] += xe; B[15] += xf;
}

static __device__ __forceinline__ uint4& operator^=(uint4& left, const uint4& right)
{
    left.x ^= right.x;
    left.y ^= right.y;
    left.z ^= right.z;
    left.w ^= right.w;
    return left;
}

static __device__ __forceinline__ void lock(int *mutex, int i)
{
    while( atomicCAS( &mutex[i], 0, 1 ) != 0 )
    {
        // keep the (slow) special function unit busy to avoid hammering
        // the memory controller with atomic operations while busy waiting
        asm volatile("{\t\n.reg .f32 tmp;\t\n"
                     "lg2.approx.f32 tmp, 0f00000000;\t\n"
                     "lg2.approx.f32 tmp, 0f00000000;\t\n}" :: );
    }
}

static __device__ __forceinline__ void unlock(int *mutex, int i)
{
    atomicExch( &mutex[i], 0 );
}

////////////////////////////////////////////////////////////////////////////////
//! Scrypt core kernel using titans to cut shared memory use in half.
//! Ideal for Kepler devices where shared memory use prevented optimal occupancy.
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
template <int WARPS_PER_BLOCK> __global__ void
scrypt_core_kernel_titanA(uint32_t *g_idata, int *mutex)
{
     // bank conflict mitigation:  +4 for alignment for uint4 in PTX >=2.0 ISA
    __shared__ uint32_t X[(WARPS_PER_BLOCK+1)/2][WU_PER_WARP][16+4];

    volatile int warpIdx        = threadIdx.x / warpSize;
    volatile int warpThread     = threadIdx.x % warpSize;
    volatile int warpIdx_2      = warpIdx/2;

    // variables supporting the large memory transaction magic
    unsigned int Y = warpThread/4;
    unsigned int Z = 4*(warpThread%4);

    // add block specific offsets
    int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_idata += 32 * offset;
    uint32_t * V = c_V[offset / WU_PER_WARP] + SCRATCH*Y + Z;

    // registers to store an entire work unit
    uint32_t B[16], C[16];

    if (warpThread == 0) lock(mutex, blockIdx.x * (WARPS_PER_BLOCK+1)/2 + warpIdx_2);

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&V[SCRATCH*wu])) = *((uint4*)(&X[warpIdx_2][wu+Y][Z])) = *((uint4*)(&g_idata[32*(wu+Y)+Z]));
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) B[idx] = X[warpIdx_2][warpThread][idx];

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&V[SCRATCH*wu+16])) = *((uint4*)(&X[warpIdx_2][wu+Y][Z])) = *((uint4*)(&g_idata[32*(wu+Y)+16+Z]));
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) C[idx] = X[warpIdx_2][warpThread][idx];

    for (int i = 1; i < 1024; i++) {

        if (warpThread == 0) unlock(mutex, blockIdx.x * (WARPS_PER_BLOCK+1)/2 + warpIdx_2);
        xor_salsa8(B, C); xor_salsa8(C, B);
        if (warpThread == 0) lock(mutex, blockIdx.x * (WARPS_PER_BLOCK+1)/2 + warpIdx_2);

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpIdx_2][warpThread][idx] = B[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)(&V[SCRATCH*wu + i*32])) = *((uint4*)(&X[warpIdx_2][wu+Y][Z]));

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpIdx_2][warpThread][idx] = C[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)(&V[SCRATCH*wu + i*32 + 16])) = *((uint4*)(&X[warpIdx_2][wu+Y][Z]));
    }
    if (warpThread == 0) unlock(mutex, blockIdx.x * (WARPS_PER_BLOCK+1)/2 + warpIdx_2);
}

template <int WARPS_PER_BLOCK> __global__ void
scrypt_core_kernel_titanB(uint32_t *g_odata, int *mutex)
{
    // bank conflict mitigation:  +4 for alignment for uint4 in PTX >=2.0 ISA
    __shared__ uint32_t X[(WARPS_PER_BLOCK+1)/2][WU_PER_WARP][16+4];

    volatile int warpIdx        = threadIdx.x / warpSize;
    volatile int warpThread     = threadIdx.x % warpSize;
    volatile int warpIdx_2      = warpIdx/2;

    // variables supporting the large memory transaction magic
    unsigned int Y = warpThread/4;
    unsigned int Z = 4*(warpThread%4);

    // add block specific offsets
    int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_odata += 32 * offset;
    uint32_t * V = c_V[offset / WU_PER_WARP] + SCRATCH*Y + Z;

    // registers to store an entire work unit
    uint32_t B[16], C[16];

    if (warpThread == 0) lock(mutex, blockIdx.x * (WARPS_PER_BLOCK+1)/2 + warpIdx_2);

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&X[warpIdx_2][wu+Y][Z])) = __ldg((uint4*)(&V[SCRATCH*wu + 1023*32]));
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) B[idx] = X[warpIdx_2][warpThread][idx];

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&X[warpIdx_2][wu+Y][Z])) = __ldg((uint4*)(&V[SCRATCH*wu + 1023*32 + 16]));
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) C[idx] = X[warpIdx_2][warpThread][idx];

    if (warpThread == 0) unlock(mutex, blockIdx.x * (WARPS_PER_BLOCK+1)/2 + warpIdx_2);
    xor_salsa8(B, C); xor_salsa8(C, B);
    if (warpThread == 0) lock(mutex, blockIdx.x * (WARPS_PER_BLOCK+1)/2 + warpIdx_2);

    for (int i = 0; i < 1024; i++) {

        X[warpIdx_2][warpThread][16] = C[0];

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpIdx_2][warpThread][idx] = B[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)(&X[warpIdx_2][wu+Y][Z])) ^= __ldg((uint4*)(&V[SCRATCH*wu + 32*(X[warpIdx_2][wu+Y][16] & 1023)]));
#pragma unroll 16
        for (int idx=0; idx < 16; idx++) B[idx] = X[warpIdx_2][warpThread][idx];

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpIdx_2][warpThread][idx] = C[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)(&X[warpIdx_2][wu+Y][Z])) ^= __ldg((uint4*)(&V[SCRATCH*wu + 32*(X[warpIdx_2][wu+Y][16] & 1023) + 16]));
#pragma unroll 16
        for (int idx=0; idx < 16; idx++) C[idx] = X[warpIdx_2][warpThread][idx];

        if (warpThread == 0) unlock(mutex, blockIdx.x * (WARPS_PER_BLOCK+1)/2 + warpIdx_2);
        xor_salsa8(B, C); xor_salsa8(C, B);
        if (warpThread == 0) lock(mutex, blockIdx.x * (WARPS_PER_BLOCK+1)/2 + warpIdx_2);
    }

#pragma unroll 16
    for (int idx=0; idx < 16; ++idx) X[warpIdx_2][warpThread][idx] = B[idx];
#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&g_odata[32*(wu+Y)+Z])) = *((uint4*)(&X[warpIdx_2][wu+Y][Z]));

#pragma unroll 16
    for (int idx=0; idx < 16; ++idx) X[warpIdx_2][warpThread][idx] = C[idx];
#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&g_odata[32*(wu+Y)+16+Z])) = *((uint4*)(&X[warpIdx_2][wu+Y][Z]));

    if (warpThread == 0) unlock(mutex, blockIdx.x * (WARPS_PER_BLOCK+1)/2 + warpIdx_2);
}
