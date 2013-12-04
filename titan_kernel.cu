//
// Kernel that runs best on Kepler (Compute 3.5) devices
//
// - makes use of 8 byte of Kepler's shared memory bank mode
// - does memory transfers with ulonglong2 vectors whereever possible
// - further halves shared memory consumption over Fermi kernel by sharing
//   the same shared memory buffers among two warps. Requires spinlocks
//   based on global atomics.
// - uses funnel shifter and __ldg() intrinsics from Compute 3.5 ISA
// - suffers from unfavorable shared memory alignment (+4 instead of +1)#
//   due to different PTX ISA alignment requirements

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

// this is a piece missing from CUDA's sm_35_intrinsics.h header
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __LDG_PTR "l"
#else
#define __LDG_PTR "r"
#endif
static __device__ __inline__ ulonglong2 __ldg(ulonglong2 *ptr) { ulonglong2 ret; asm volatile ("ld.global.nc.v2.u64 {%0,%1}, [%2];" : "=l"(ret.x), "=l"(ret.y) : __LDG_PTR (ptr)); return ret; }

// forward references
template <int WARPS_PER_BLOCK> __global__ void titan_scrypt_core_kernelA(uint32_t *g_idata);
template <int WARPS_PER_BLOCK> __global__ void titan_scrypt_core_kernelB(uint32_t *g_odata);

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
        case 1: titan_scrypt_core_kernelA<1><<< grid, threads, 0, stream >>>(d_idata); break;
        case 2: titan_scrypt_core_kernelA<2><<< grid, threads, 0, stream >>>(d_idata); break;
        case 3: titan_scrypt_core_kernelA<3><<< grid, threads, 0, stream >>>(d_idata); break;
        case 4: titan_scrypt_core_kernelA<4><<< grid, threads, 0, stream >>>(d_idata); break;
        case 5: titan_scrypt_core_kernelA<5><<< grid, threads, 0, stream >>>(d_idata); break;
        case 6: titan_scrypt_core_kernelA<6><<< grid, threads, 0, stream >>>(d_idata); break;
        case 7: titan_scrypt_core_kernelA<7><<< grid, threads, 0, stream >>>(d_idata); break;
        case 8: titan_scrypt_core_kernelA<8><<< grid, threads, 0, stream >>>(d_idata); break;
        case 9: titan_scrypt_core_kernelA<9><<< grid, threads, 0, stream >>>(d_idata); break;
        case 10: titan_scrypt_core_kernelA<10><<< grid, threads, 0, stream >>>(d_idata); break;
        case 11: titan_scrypt_core_kernelA<11><<< grid, threads, 0, stream >>>(d_idata); break;
        case 12: titan_scrypt_core_kernelA<12><<< grid, threads, 0, stream >>>(d_idata); break;
        case 13: titan_scrypt_core_kernelA<13><<< grid, threads, 0, stream >>>(d_idata); break;
        case 14: titan_scrypt_core_kernelA<14><<< grid, threads, 0, stream >>>(d_idata); break;
        case 15: titan_scrypt_core_kernelA<15><<< grid, threads, 0, stream >>>(d_idata); break;
        case 16: titan_scrypt_core_kernelA<16><<< grid, threads, 0, stream >>>(d_idata); break;
        case 17: titan_scrypt_core_kernelA<17><<< grid, threads, 0, stream >>>(d_idata); break;
        case 18: titan_scrypt_core_kernelA<18><<< grid, threads, 0, stream >>>(d_idata); break;
        case 19: titan_scrypt_core_kernelA<19><<< grid, threads, 0, stream >>>(d_idata); break;
        case 20: titan_scrypt_core_kernelA<20><<< grid, threads, 0, stream >>>(d_idata); break;
        case 21: titan_scrypt_core_kernelA<21><<< grid, threads, 0, stream >>>(d_idata); break;
        case 22: titan_scrypt_core_kernelA<22><<< grid, threads, 0, stream >>>(d_idata); break;
        case 23: titan_scrypt_core_kernelA<23><<< grid, threads, 0, stream >>>(d_idata); break;
        case 24: titan_scrypt_core_kernelA<24><<< grid, threads, 0, stream >>>(d_idata); break;
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
        case 1: titan_scrypt_core_kernelB<1><<< grid, threads, 0, stream >>>(d_odata); break;
        case 2: titan_scrypt_core_kernelB<2><<< grid, threads, 0, stream >>>(d_odata); break;
        case 3: titan_scrypt_core_kernelB<3><<< grid, threads, 0, stream >>>(d_odata); break;
        case 4: titan_scrypt_core_kernelB<4><<< grid, threads, 0, stream >>>(d_odata); break;
        case 5: titan_scrypt_core_kernelB<5><<< grid, threads, 0, stream >>>(d_odata); break;
        case 6: titan_scrypt_core_kernelB<6><<< grid, threads, 0, stream >>>(d_odata); break;
        case 7: titan_scrypt_core_kernelB<7><<< grid, threads, 0, stream >>>(d_odata); break;
        case 8: titan_scrypt_core_kernelB<8><<< grid, threads, 0, stream >>>(d_odata); break;
        case 9: titan_scrypt_core_kernelB<9><<< grid, threads, 0, stream >>>(d_odata); break;
        case 10: titan_scrypt_core_kernelB<10><<< grid, threads, 0, stream >>>(d_odata); break;
        case 11: titan_scrypt_core_kernelB<11><<< grid, threads, 0, stream >>>(d_odata); break;
        case 12: titan_scrypt_core_kernelB<12><<< grid, threads, 0, stream >>>(d_odata); break;
        case 13: titan_scrypt_core_kernelB<13><<< grid, threads, 0, stream >>>(d_odata); break;
        case 14: titan_scrypt_core_kernelB<14><<< grid, threads, 0, stream >>>(d_odata); break;
        case 15: titan_scrypt_core_kernelB<15><<< grid, threads, 0, stream >>>(d_odata); break;
        case 16: titan_scrypt_core_kernelB<16><<< grid, threads, 0, stream >>>(d_odata); break;
        case 17: titan_scrypt_core_kernelB<17><<< grid, threads, 0, stream >>>(d_odata); break;
        case 18: titan_scrypt_core_kernelB<18><<< grid, threads, 0, stream >>>(d_odata); break;
        case 19: titan_scrypt_core_kernelB<19><<< grid, threads, 0, stream >>>(d_odata); break;
        case 20: titan_scrypt_core_kernelB<20><<< grid, threads, 0, stream >>>(d_odata); break;
        case 21: titan_scrypt_core_kernelB<21><<< grid, threads, 0, stream >>>(d_odata); break;
        case 22: titan_scrypt_core_kernelB<22><<< grid, threads, 0, stream >>>(d_odata); break;
        case 23: titan_scrypt_core_kernelB<23><<< grid, threads, 0, stream >>>(d_odata); break;
        case 24: titan_scrypt_core_kernelB<24><<< grid, threads, 0, stream >>>(d_odata); break;
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

static __device__ void xor_salsa8(uint4 *B, uint4 *C)
{
	uint32_t x[16];
	x[0]=(B[0].x ^= C[0].x);
	x[1]=(B[0].y ^= C[0].y);
	x[2]=(B[0].z ^= C[0].z);
	x[3]=(B[0].w ^= C[0].w);
	x[4]=(B[1].x ^= C[1].x);
	x[5]=(B[1].y ^= C[1].y);
	x[6]=(B[1].z ^= C[1].z);
	x[7]=(B[1].w ^= C[1].w);
	x[8]=(B[2].x ^= C[2].x);
	x[9]=(B[2].y ^= C[2].y);
	x[10]=(B[2].z ^= C[2].z);
	x[11]=(B[2].w ^= C[2].w);
	x[12]=(B[3].x ^= C[3].x);
	x[13]=(B[3].y ^= C[3].y);
	x[14]=(B[3].z ^= C[3].z);
	x[15]=(B[3].w ^= C[3].w);

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

    B[0].x += x[0]; B[0].y += x[1]; B[0].z += x[2];  B[0].w += x[3];  B[1].x += x[4];  B[1].y += x[5];  B[1].z += x[6];  B[1].w += x[7];
    B[2].x += x[8]; B[2].y += x[9]; B[2].z += x[10]; B[2].w += x[11]; B[3].x += x[12]; B[3].y += x[13]; B[3].z += x[14]; B[3].w += x[15];
}

static __device__ __forceinline__ uint4& operator^=(uint4& left, const uint4& right)
{
    left.x ^= right.x;
    left.y ^= right.y;
    left.z ^= right.z;
    left.w ^= right.w;
    return left;
}

static __device__ __forceinline__ void lock(int *mutex, const int i)
{
    while( atomicCAS( &mutex[i], 0, 1 ) != 0 );
}

static __device__ __forceinline__ void unlock(int *mutex, const int i)
{
    atomicExch( &mutex[i], 0 );
}

// Number of shared memory buffers and mapping between warp and buffer
// I made this a #define so I could play with different parameters.
#define SHARED_BUFFERS  (WARPS_PER_BLOCK+1)/2
#define BUFFER_MAPPING  warpIdx/2

////////////////////////////////////////////////////////////////////////////////
//! Scrypt core kernel using spinlocks to cut shared memory use in half.
//! Ideal for Kepler devices where shared memory use prevented optimal occupancy.
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
template <int WARPS_PER_BLOCK> __global__ void
titan_scrypt_core_kernelA(uint32_t *g_idata)
{
     // bank conflict mitigation:  +4 for alignment for ulonglong2 in PTX >=2.0 ISA
    __shared__ uint32_t X[SHARED_BUFFERS][WU_PER_WARP][16+4];
    __shared__ int L[SHARED_BUFFERS];

    int warpIdx        = threadIdx.x / warpSize;
    int warpThread     = threadIdx.x % warpSize;

    // variables supporting the large memory transaction magic
    unsigned int Y = warpThread/4;
    unsigned int Z = 4*(warpThread%4);

    // add block specific offsets
    int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_idata += 32 * offset;
    uint32_t * V = c_V[offset / WU_PER_WARP] + SCRATCH*Y + Z;

    // registers to store an entire work unit
    uint4 B[4], C[4];

    volatile int lockIdx = BUFFER_MAPPING;
    uint32_t ((*XB)[16+4]) = (uint32_t (*)[16+4])&X[lockIdx][Y][Z];
    uint32_t *XX = X[lockIdx][warpThread];

    if (threadIdx.x < SHARED_BUFFERS) L[threadIdx.x] = 0;
    __syncthreads();
    if (warpThread == 0) lock(L, lockIdx);

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((ulonglong2*)(&V[SCRATCH*wu])) = *((ulonglong2*)XB[wu]) = *((ulonglong2*)(&g_idata[32*(wu+Y)+Z]));
#pragma unroll 4
    for (int idx=0; idx < 4; idx++) B[idx] = *((uint4*)&XX[4*idx]);

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((ulonglong2*)(&V[SCRATCH*wu+16])) = *((ulonglong2*)XB[wu]) = *((ulonglong2*)(&g_idata[32*(wu+Y)+16+Z]));
#pragma unroll 4
    for (int idx=0; idx < 4; idx++) C[idx] = *((uint4*)&XX[4*idx]);

    for (int i = 1; i < 1024; i++) {

        if (warpThread == 0) unlock(L, lockIdx);
        xor_salsa8(B, C); xor_salsa8(C, B);
        if (warpThread == 0) lock(L, lockIdx);

#pragma unroll 4
        for (int idx=0; idx < 4; ++idx) *((uint4*)&XX[4*idx]) = B[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((ulonglong2*)(&V[SCRATCH*wu + i*32])) = *((ulonglong2*)XB[wu]);

#pragma unroll 4
        for (int idx=0; idx < 4; ++idx) *((uint4*)&XX[4*idx]) = C[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((ulonglong2*)(&V[SCRATCH*wu + i*32 + 16])) = *((ulonglong2*)XB[wu]);
    }
    if (warpThread == 0) unlock(L, lockIdx);
}

template <int WARPS_PER_BLOCK> __global__ void
titan_scrypt_core_kernelB(uint32_t *g_odata)
{
    // bank conflict mitigation:  +4 for alignment for ulonglong2 in PTX >=2.0 ISA
    __shared__ uint32_t X[SHARED_BUFFERS][WU_PER_WARP][16+4];
    __shared__ int L[SHARED_BUFFERS];

    int warpIdx        = threadIdx.x / warpSize;
    int warpThread     = threadIdx.x % warpSize;

    // variables supporting the large memory transaction magic
    unsigned int Y = warpThread/4;
    unsigned int Z = 4*(warpThread%4);

    // add block specific offsets
    int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_odata += 32 * offset;
    uint32_t * V = c_V[offset / WU_PER_WARP] + SCRATCH*Y + Z;

    // registers to store an entire work unit
    uint4 B[4], C[4];

    volatile int lockIdx = BUFFER_MAPPING;
    uint32_t ((*XB)[16+4]) = (uint32_t (*)[16+4])&X[lockIdx][Y][Z];
    uint32_t *XX = X[lockIdx][warpThread];

    if (threadIdx.x < SHARED_BUFFERS) L[threadIdx.x] = 0;
    __syncthreads();
    if (warpThread == 0) lock(L, lockIdx);

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((ulonglong2*)XB[wu]) = __ldg((ulonglong2*)(&V[SCRATCH*wu + 1023*32]));
#pragma unroll 4
    for (int idx=0; idx < 4; idx++) B[idx] = *((uint4*)&XX[4*idx]);

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((ulonglong2*)XB[wu]) = __ldg((ulonglong2*)(&V[SCRATCH*wu + 1023*32 + 16]));
#pragma unroll 4
    for (int idx=0; idx < 4; idx++) C[idx] = *((uint4*)&XX[4*idx]);

    if (warpThread == 0) unlock(L, lockIdx);
    xor_salsa8(B, C); xor_salsa8(C, B);
    if (warpThread == 0) lock(L, lockIdx);

    for (int i = 0; i < 1024; i++) {

        XX[16] = 32 * (C[0].x & 1023);

#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((ulonglong2*)XB[wu]) = __ldg((ulonglong2*)(&V[SCRATCH*wu + XB[wu][16-Z]]));
#pragma unroll 4
        for (int idx=0; idx < 4; idx++) B[idx] ^= *((uint4*)&XX[4*idx]);

#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((ulonglong2*)XB[wu]) = __ldg((ulonglong2*)(&V[SCRATCH*wu + XB[wu][16-Z] + 16]));
#pragma unroll 4
        for (int idx=0; idx < 4; idx++) C[idx] ^= *((uint4*)&XX[4*idx]);

        if (warpThread == 0) unlock(L, lockIdx);
        xor_salsa8(B, C); xor_salsa8(C, B);
        if (warpThread == 0) lock(L, lockIdx);
    }

#pragma unroll 4
    for (int idx=0; idx < 4; ++idx) *((uint4*)&XX[4*idx]) = B[idx];
#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((ulonglong2*)(&g_odata[32*(wu+Y)+Z])) = *((ulonglong2*)XB[wu]);

#pragma unroll 4
    for (int idx=0; idx < 4; ++idx) *((uint4*)&XX[4*idx]) = C[idx];
#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((ulonglong2*)(&g_odata[32*(wu+Y)+16+Z])) = *((ulonglong2*)XB[wu]);

    if (warpThread == 0) unlock(L, lockIdx);
}
