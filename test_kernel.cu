//
// Experimental Kernel for Kepler (Compute 3.5) devices
//
// Eliminates shared memory entirely, uses warp shuffle instructions instead
// based on a technique found in this blog posting by Allan MacKinnon:
// http://www.pixel.io/blog/2013/4/7/fast-matrix-transposition-without-shuffling-or-shared-memory.html
//
// Does not yet run as fast as the shared memory based kernel, but there may
// be further room for optimization! (417 kHash/s vs 450 kHash/s for T kernel)
// The card also seems to run hotter, running into its thermal limits sooner.
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

#include "test_kernel.h"

// grab lane ID
static __device__ __inline__ unsigned int __laneId() { unsigned int laneId; asm( "mov.u32 %0, %%laneid;" : "=r"( laneId ) ); return laneId; }

// forward references
__global__ void test_scrypt_core_kernelA(uint32_t *g_idata);
__global__ void test_scrypt_core_kernelB(uint32_t *g_odata);

// scratchbuf constants (pointers to scratch buffer for each work unit)
__constant__ uint32_t* c_V[1024];

TestKernel::TestKernel() : KernelInterface()
{
}

void TestKernel::set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_V, h_V, MAXWARPS*sizeof(uint32_t*), 0, cudaMemcpyHostToDevice));
}

bool TestKernel::run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, bool interactive, bool benchmark, int texture_cache)
{
    bool success = true;

    // clear CUDA's error variable
    cudaGetLastError();

    // First phase: Sequential writes to scratchpad.

    test_scrypt_core_kernelA<<< grid, threads, 0, stream >>>(d_idata);

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

    test_scrypt_core_kernelB<<< grid, threads, 0, stream >>>(d_odata);

    // catch any kernel launch failures
    if (cudaPeekAtLastError() != cudaSuccess) success = false;

    return success;
}

static __device__ uint4& operator^=(uint4& left, const uint4& right)
{
    left.x ^= right.x;
    left.y ^= right.y;
    left.z ^= right.z;
    left.w ^= right.w;
    return left;
}

__device__ __forceinline__ uint4 __shfl(const uint4 val, unsigned int lane)
{
    return make_uint4(
        (unsigned int)__shfl((int)val.x, lane),
        (unsigned int)__shfl((int)val.y, lane),
        (unsigned int)__shfl((int)val.z, lane),
        (unsigned int)__shfl((int)val.w, lane));
}

__device__ __forceinline__ void __swap(uint4 &a, uint4 &b)
{
//    uint4 t = b; b = a; a = t;
    uint32_t t;
    t=a.x; a.x=b.x; b.x=t;
    t=a.y; a.y=b.y; b.y=t;
    t=a.z; a.z=b.z; b.z=t;
    t=a.w; a.w=b.w; b.w=t;
}

__device__ __forceinline__ void __transposed_write(uint4 (&S)[4], uint4 *D, int spacing=1)
{
    unsigned int laneId = __laneId();

    unsigned int lane4 = laneId%4;
    unsigned int tile  = laneId/4;
    unsigned int tile4 = tile*4;

    unsigned int rot3 = tile4+(lane4+3)%4;
    unsigned int rot2 = tile4+(lane4+2)%4;
    unsigned int rot1 = tile4+(lane4+1)%4;

    // rotate
    S[1] = __shfl(S[1], rot3);
    S[2] = __shfl(S[2], rot2);
    S[3] = __shfl(S[3], rot1);

    // exchange
    if (lane4 >= 2) { __swap(S[0], S[2]); __swap(S[1], S[3]); }

    // select + write
    D[spacing*2*(16*tile   )+ lane4     ] = (laneId % 2 == 0) ? S[0] : S[1];
    D[spacing*2*(16*tile+4 )+(lane4+3)%4] = (laneId % 2 == 0) ? S[3] : S[0];
    D[spacing*2*(16*tile+8 )+(lane4+2)%4] = (laneId % 2 == 0) ? S[2] : S[3];
    D[spacing*2*(16*tile+12)+(lane4+1)%4] = (laneId % 2 == 0) ? S[1] : S[2];

    // undo exchange
    if (lane4 >= 2) { __swap(S[0], S[2]); __swap(S[1], S[3]); }

    // undo rotate
    S[1] = __shfl(S[1], rot1);
    S[2] = __shfl(S[2], rot2);
    S[3] = __shfl(S[3], rot3);
}

__device__ __forceinline__ void __transposed_read(uint4 *S, uint4 (&D)[4], int spacing=1)
{
    unsigned int laneId = __laneId();

    unsigned int lane4 = laneId%4;
    unsigned int tile  = laneId/4;
    unsigned int tile4 = tile*4;

    unsigned int rot3 = tile4+(lane4+3)%4;
    unsigned int rot2 = tile4+(lane4+2)%4;
    unsigned int rot1 = tile4+(lane4+1)%4;

    // read and select
    uint4 tmp; 
    tmp = __ldg(&S[spacing*2*(16*tile   )+ lane4     ]); if (laneId % 2 == 0) D[0] = tmp; else D[1] = tmp;
    tmp = __ldg(&S[spacing*2*(16*tile+4 )+(lane4+3)%4]); if (laneId % 2 == 0) D[3] = tmp; else D[0] = tmp;
    tmp = __ldg(&S[spacing*2*(16*tile+8 )+(lane4+2)%4]); if (laneId % 2 == 0) D[2] = tmp; else D[3] = tmp;
    tmp = __ldg(&S[spacing*2*(16*tile+12)+(lane4+1)%4]); if (laneId % 2 == 0) D[1] = tmp; else D[2] = tmp;

    // undo exchange
    if (lane4 >= 2) { __swap(D[0], D[2]); __swap(D[1], D[3]); }

    // undo rotate
    D[1] = __shfl(D[1], rot1);
    D[2] = __shfl(D[2], rot2);
    D[3] = __shfl(D[3], rot3);
}

__device__ __forceinline__ void __transposed_xor(uint4 *S, uint4 (&D)[4], int spacing=1, int row=0)
{
    unsigned int laneId = __laneId();

    unsigned int lane4 = laneId%4;
    unsigned int tile  = laneId/4;
    unsigned int tile4 = tile*4;

    unsigned int rot3 = tile4+(lane4+3)%4;
    unsigned int rot2 = tile4+(lane4+2)%4;
    unsigned int rot1 = tile4+(lane4+1)%4;

    // rotate
    D[1] = __shfl(D[1], rot3);
    D[2] = __shfl(D[2], rot2);
    D[3] = __shfl(D[3], rot1);

    // exchange
    if (lane4 >= 2) { __swap(D[0], D[2]); __swap(D[1], D[3]); }

    // read and select
    uint4 tmp; 
    tmp = __ldg(&S[spacing*2*(16*tile   )+ lane4     +8*__shfl(row,tile4  )]); if (laneId % 2 == 0) D[0] ^= tmp; else D[1] ^= tmp;
    tmp = __ldg(&S[spacing*2*(16*tile+4 )+(lane4+3)%4+8*__shfl(row,tile4+1)]); if (laneId % 2 == 0) D[3] ^= tmp; else D[0] ^= tmp;
    tmp = __ldg(&S[spacing*2*(16*tile+8 )+(lane4+2)%4+8*__shfl(row,tile4+2)]); if (laneId % 2 == 0) D[2] ^= tmp; else D[3] ^= tmp;
    tmp = __ldg(&S[spacing*2*(16*tile+12)+(lane4+1)%4+8*__shfl(row,tile4+3)]); if (laneId % 2 == 0) D[1] ^= tmp; else D[2] ^= tmp;

    // undo exchange
    if (lane4 >= 2) { __swap(D[0], D[2]); __swap(D[1], D[3]); }

    // undo rotate
    D[1] = __shfl(D[1], rot1);
    D[2] = __shfl(D[2], rot2);
    D[3] = __shfl(D[3], rot3);
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

////////////////////////////////////////////////////////////////////////////////
//! Experimental Scrypt core kernel for Titan devices.
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void test_scrypt_core_kernelA(uint32_t *g_idata)
{
    // add warp specific offsets
    int offset = blockIdx.x * blockDim.x + threadIdx.x / warpSize * warpSize;
    g_idata += 32 * offset;
    uint32_t * V = c_V[offset / warpSize];

    // registers to store an entire work unit
    uint4 B[4], C[4];

    __transposed_read((uint4*)(g_idata)   , B, 1);
    __transposed_read((uint4*)(g_idata+16), C, 1);

    __transposed_write(B, (uint4*)V, 1024); V+=16;
    __transposed_write(C, (uint4*)V, 1024); V+=16;

    for (int i = 1; i < 1024; i++) {

        xor_salsa8(B, C); xor_salsa8(C, B);

        __transposed_write(B, (uint4*)V, 1024); V+=16;
        __transposed_write(C, (uint4*)V, 1024); V+=16;
    }
}

__global__ void test_scrypt_core_kernelB(uint32_t *g_odata)
{
    // add warp specific offsets
    int offset = blockIdx.x * blockDim.x + threadIdx.x / warpSize * warpSize;
    g_odata += 32 * offset;
    uint32_t * V = c_V[offset / warpSize];

    // registers to store an entire work unit
    uint4 B[4], C[4];

    __transposed_read((uint4*)(V+1023*32),    B, 1024);
    __transposed_read((uint4*)(V+1023*32+16), C, 1024);

    xor_salsa8(B, C); xor_salsa8(C, B);

    for (int i = 0; i < 1024; i++) {

        __transposed_xor((uint4*)(V),    B, 1024, (C[0].x & 1023));
        __transposed_xor((uint4*)(V+16), C, 1024, (C[0].x & 1023));

        xor_salsa8(B, C); xor_salsa8(C, B);
    }

    __transposed_write(B, (uint4*)(g_odata)   , 1);
    __transposed_write(C, (uint4*)(g_odata+16), 1);
}
