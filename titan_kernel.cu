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
        case 17: scrypt_core_kernel_titanA<17><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 18: scrypt_core_kernel_titanA<18><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 19: scrypt_core_kernel_titanA<19><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 20: scrypt_core_kernel_titanA<20><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 21: scrypt_core_kernel_titanA<21><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 22: scrypt_core_kernel_titanA<22><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 23: scrypt_core_kernel_titanA<23><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
        case 24: scrypt_core_kernel_titanA<24><<< grid, threads, 0, stream >>>(d_idata, mutex); break;
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
        case 17: scrypt_core_kernel_titanB<17><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 18: scrypt_core_kernel_titanB<18><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 19: scrypt_core_kernel_titanB<19><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 20: scrypt_core_kernel_titanB<20><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 21: scrypt_core_kernel_titanB<21><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 22: scrypt_core_kernel_titanB<22><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 23: scrypt_core_kernel_titanB<23><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        case 24: scrypt_core_kernel_titanB<24><<< grid, threads, 0, stream >>>(d_odata, mutex); break;
        default: success = false; break;
    }

    // catch any kernel launch failures
    if (cudaPeekAtLastError() != cudaSuccess) success = false;

    return success;
}

#define ROTL(a, b) __funnelshift_l( a, a, b );

#define ROTL7(a0,a1,a2,a3,a00,a10,a20,a30){\
a0^=ROTL(a00, 7); a1^=ROTL(a10, 7); a2^=ROTL(a20, 7); a3^=ROTL(a30, 7);\
};\

#define ROTL9(a0,a1,a2,a3,a00,a10,a20,a30){\
a0^=ROTL(a00, 9); a1^=ROTL(a10, 9); a2^=ROTL(a20, 9); a3^=ROTL(a30, 9);\
};\

#define ROTL13(a0,a1,a2,a3,a00,a10,a20,a30){\
a0^=ROTL(a00, 13); a1^=ROTL(a10, 13); a2^=ROTL(a20, 13); a3^=ROTL(a30, 13);\
};\

#define ROTL18(a0,a1,a2,a3,a00,a10,a20,a30){\
a0^=ROTL(a00, 18); a1^=ROTL(a10, 18); a2^=ROTL(a20, 18); a3^=ROTL(a30, 18);\
};\

static __device__ void xor_salsa8(uint32_t *B,uint32_t *C)
{
	uint32_t x[16];
	x[0]=(B[0] ^= C[0]);
	x[1]=(B[1] ^= C[1]);
	x[2]=(B[2] ^= C[2]);
	x[3]=(B[3] ^= C[3]);
	x[4]=(B[4] ^= C[4]);
	x[5]=(B[5] ^= C[5]);
	x[6]=(B[6] ^= C[6]);
	x[7]=(B[7] ^= C[7]);
	x[8]=(B[8] ^= C[8]);
	x[9]=(B[9] ^= C[9]);
	x[10]=(B[10] ^= C[10]);
	x[11]=(B[11] ^= C[11]);
	x[12]=(B[12] ^= C[12]);
	x[13]=(B[13] ^= C[13]);
	x[14]=(B[14] ^= C[14]);
	x[15]=(B[15] ^= C[15]);

    /* Operate on columns. */
	ROTL7(x[4],x[9],x[14],x[3],x[0]+x[12],x[1]+x[5],x[6]+x[10],x[11]+x[15]);
	ROTL9(x[8],x[13],x[2],x[7],x[0]+x[4],x[5]+x[9],x[10]+x[14],x[3]+x[15]);
	ROTL13(x[12],x[1],x[6],x[11],x[4]+x[8],x[9]+x[13],x[2]+x[14],x[3]+x[7]);
	ROTL18(x[0],x[5],x[10],x[15],x[8]+x[12],x[1]+x[13],x[2]+x[6],x[7]+x[11]);

    /* Operate on rows. */
	ROTL7(x[1],x[6],x[11],x[12],x[0]+x[3],x[4]+x[5],x[9]+x[10],x[14]+x[15]);
	ROTL9(x[2],x[7],x[8],x[13],x[0]+x[1],x[5]+x[6],x[10]+x[11],x[12]+x[15]);
	ROTL13(x[3],x[4],x[9],x[14],x[1]+x[2],x[6]+x[7],x[8]+x[11],x[12]+x[13]);
	ROTL18(x[0],x[5],x[10],x[15],x[2]+x[3],x[4]+x[7],x[8]+x[9],x[13]+x[14]);

    /* Operate on columns. */
	ROTL7(x[4],x[9],x[14],x[3],x[0]+x[12],x[1]+x[5],x[6]+x[10],x[11]+x[15]);
	ROTL9(x[8],x[13],x[2],x[7],x[0]+x[4],x[5]+x[9],x[10]+x[14],x[3]+x[15]);
	ROTL13(x[12],x[1],x[6],x[11],x[4]+x[8],x[9]+x[13],x[2]+x[14],x[3]+x[7]);
	ROTL18(x[0],x[5],x[10],x[15],x[8]+x[12],x[1]+x[13],x[2]+x[6],x[7]+x[11]);

    /* Operate on rows. */
	ROTL7(x[1],x[6],x[11],x[12],x[0]+x[3],x[4]+x[5],x[9]+x[10],x[14]+x[15]);
	ROTL9(x[2],x[7],x[8],x[13],x[0]+x[1],x[5]+x[6],x[10]+x[11],x[12]+x[15]);
	ROTL13(x[3],x[4],x[9],x[14],x[1]+x[2],x[6]+x[7],x[8]+x[11],x[12]+x[13]);
	ROTL18(x[0],x[5],x[10],x[15],x[2]+x[3],x[4]+x[7],x[8]+x[9],x[13]+x[14]);

    /* Operate on columns. */
	ROTL7(x[4],x[9],x[14],x[3],x[0]+x[12],x[1]+x[5],x[6]+x[10],x[11]+x[15]);
	ROTL9(x[8],x[13],x[2],x[7],x[0]+x[4],x[5]+x[9],x[10]+x[14],x[3]+x[15]);
	ROTL13(x[12],x[1],x[6],x[11],x[4]+x[8],x[9]+x[13],x[2]+x[14],x[3]+x[7]);
	ROTL18(x[0],x[5],x[10],x[15],x[8]+x[12],x[1]+x[13],x[2]+x[6],x[7]+x[11]);

    /* Operate on rows. */
	ROTL7(x[1],x[6],x[11],x[12],x[0]+x[3],x[4]+x[5],x[9]+x[10],x[14]+x[15]);
	ROTL9(x[2],x[7],x[8],x[13],x[0]+x[1],x[5]+x[6],x[10]+x[11],x[12]+x[15]);
	ROTL13(x[3],x[4],x[9],x[14],x[1]+x[2],x[6]+x[7],x[8]+x[11],x[12]+x[13]);
	ROTL18(x[0],x[5],x[10],x[15],x[2]+x[3],x[4]+x[7],x[8]+x[9],x[13]+x[14]);

    /* Operate on columns. */
	ROTL7(x[4],x[9],x[14],x[3],x[0]+x[12],x[1]+x[5],x[6]+x[10],x[11]+x[15]);
	ROTL9(x[8],x[13],x[2],x[7],x[0]+x[4],x[5]+x[9],x[10]+x[14],x[3]+x[15]);
	ROTL13(x[12],x[1],x[6],x[11],x[4]+x[8],x[9]+x[13],x[2]+x[14],x[3]+x[7]);
	ROTL18(x[0],x[5],x[10],x[15],x[8]+x[12],x[1]+x[13],x[2]+x[6],x[7]+x[11]);

    /* Operate on rows. */
	ROTL7(x[1],x[6],x[11],x[12],x[0]+x[3],x[4]+x[5],x[9]+x[10],x[14]+x[15]);
	ROTL9(x[2],x[7],x[8],x[13],x[0]+x[1],x[5]+x[6],x[10]+x[11],x[12]+x[15]);
	ROTL13(x[3],x[4],x[9],x[14],x[1]+x[2],x[6]+x[7],x[8]+x[11],x[12]+x[13]);
	ROTL18(x[0],x[5],x[10],x[15],x[2]+x[3],x[4]+x[7],x[8]+x[9],x[13]+x[14]);

    B[ 0] += x[0]; B[ 1] += x[1]; B[ 2] += x[2]; B[ 3] += x[3]; B[ 4] += x[4]; B[ 5] += x[5]; B[ 6] += x[6]; B[ 7] += x[7];
    B[ 8] += x[8]; B[ 9] += x[9]; B[10] += x[10]; B[11] += x[11]; B[12] += x[12]; B[13] += x[13]; B[14] += x[14]; B[15] += x[15];
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
