//
// Kernel that runs best on Fermi devices
//
// NOTE: compile this .cu module for compute_10,sm_10 with --maxrregcount=124
//

#ifdef WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include <cuda.h>

#include "fermi_kernel.h"

#if WIN32
#ifdef _WIN64
#define _64BIT_ALIGN 1
#else
#define _64BIT_ALIGN 0
#endif
#else
#if __x86_64__
#define _64BIT_ALIGN 1
#else
#define _64BIT_ALIGN 0
#endif
#endif

// forward references
template <int WARPS_PER_BLOCK> __global__ void scrypt_core_kernelA(uint32_t *g_idata);
template <int WARPS_PER_BLOCK> __global__ void scrypt_core_kernelB(uint32_t *g_odata);
template <int WARPS_PER_BLOCK, int TEX_DIM> __global__ void scrypt_core_kernelB_tex(uint32_t *g_odata);

// scratchbuf constants (pointers to scratch buffer for each work unit)
__constant__ uint32_t* c_V[1024];

// using texture references for the "tex" variants of the B kernels
texture<uint4, 1, cudaReadModeElementType> texRef1D_4_V;
texture<uint4, 2, cudaReadModeElementType> texRef2D_4_V;

FermiKernel::FermiKernel() : KernelInterface()
{
}

bool FermiKernel::bindtexture_1D(uint32_t *d_V, size_t size)
{
    cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc<uint4>();
    texRef1D_4_V.normalized = 0;
    texRef1D_4_V.filterMode = cudaFilterModePoint;
    texRef1D_4_V.addressMode[0] = cudaAddressModeClamp;
    checkCudaErrors(cudaBindTexture(NULL, &texRef1D_4_V, d_V, &channelDesc4, size));
    return true;
}

bool FermiKernel::bindtexture_2D(uint32_t *d_V, int width, int height, size_t pitch)
{
    cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc<uint4>();
    texRef2D_4_V.normalized = 0;
    texRef2D_4_V.filterMode = cudaFilterModePoint;
    texRef2D_4_V.addressMode[0] = cudaAddressModeClamp;
    texRef2D_4_V.addressMode[1] = cudaAddressModeClamp;
    checkCudaErrors(cudaBindTexture2D(NULL, &texRef2D_4_V, d_V, &channelDesc4, width, height, pitch));
    return true;
}

bool FermiKernel::unbindtexture_1D()
{
    checkCudaErrors(cudaUnbindTexture(texRef1D_4_V));
    return true;
}

bool FermiKernel::unbindtexture_2D()
{
    checkCudaErrors(cudaUnbindTexture(texRef2D_4_V));
    return true;
}

void FermiKernel::set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_V, h_V, MAXWARPS*sizeof(uint32_t*), 0, cudaMemcpyHostToDevice));
}

bool FermiKernel::run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, int *mutex, bool interactive, bool benchmark, int texture_cache)
{
    bool success = true;

    // clear CUDA's error variable
    cudaGetLastError();

    // First phase: Sequential writes to scratchpad.

    switch (WARPS_PER_BLOCK) {
            case 1: scrypt_core_kernelA<1><<< grid, threads, 0, stream >>>(d_idata); break;
            case 2: scrypt_core_kernelA<2><<< grid, threads, 0, stream >>>(d_idata); break;
            case 3: scrypt_core_kernelA<3><<< grid, threads, 0, stream >>>(d_idata); break;
            case 4: scrypt_core_kernelA<4><<< grid, threads, 0, stream >>>(d_idata); break;
            case 5: scrypt_core_kernelA<5><<< grid, threads, 0, stream >>>(d_idata); break;
            case 6: scrypt_core_kernelA<6><<< grid, threads, 0, stream >>>(d_idata); break;
            case 7: scrypt_core_kernelA<7><<< grid, threads, 0, stream >>>(d_idata); break;
            case 8: scrypt_core_kernelA<8><<< grid, threads, 0, stream >>>(d_idata); break;
            case 9: scrypt_core_kernelA<9><<< grid, threads, 0, stream >>>(d_idata); break;
            case 10: scrypt_core_kernelA<10><<< grid, threads, 0, stream >>>(d_idata); break;
            case 11: scrypt_core_kernelA<11><<< grid, threads, 0, stream >>>(d_idata); break;
            case 12: scrypt_core_kernelA<12><<< grid, threads, 0, stream >>>(d_idata); break;
            case 13: scrypt_core_kernelA<13><<< grid, threads, 0, stream >>>(d_idata); break;
            case 14: scrypt_core_kernelA<14><<< grid, threads, 0, stream >>>(d_idata); break;
            case 15: scrypt_core_kernelA<15><<< grid, threads, 0, stream >>>(d_idata); break;
            case 16: scrypt_core_kernelA<16><<< grid, threads, 0, stream >>>(d_idata); break;
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

    if (texture_cache)
    {
        if (texture_cache == 1)
        {
            switch (WARPS_PER_BLOCK) {
                    case 1: scrypt_core_kernelB_tex<1,1><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 2: scrypt_core_kernelB_tex<2,1><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 3: scrypt_core_kernelB_tex<3,1><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 4: scrypt_core_kernelB_tex<4,1><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 5: scrypt_core_kernelB_tex<5,1><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 6: scrypt_core_kernelB_tex<6,1><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 7: scrypt_core_kernelB_tex<7,1><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 8: scrypt_core_kernelB_tex<8,1><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 9: scrypt_core_kernelB_tex<9,1><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 10: scrypt_core_kernelB_tex<10,1><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 11: scrypt_core_kernelB_tex<11,1><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 12: scrypt_core_kernelB_tex<12,1><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 13: scrypt_core_kernelB_tex<13,1><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 14: scrypt_core_kernelB_tex<14,1><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 15: scrypt_core_kernelB_tex<15,1><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 16: scrypt_core_kernelB_tex<16,1><<< grid, threads, 0, stream >>>(d_odata); break;
                default: success = false; break;
            }
        }
        else if (texture_cache == 2)
        {
            switch (WARPS_PER_BLOCK) {
                    case 1: scrypt_core_kernelB_tex<1,2><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 2: scrypt_core_kernelB_tex<2,2><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 3: scrypt_core_kernelB_tex<3,2><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 4: scrypt_core_kernelB_tex<4,2><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 5: scrypt_core_kernelB_tex<5,2><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 6: scrypt_core_kernelB_tex<6,2><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 7: scrypt_core_kernelB_tex<7,2><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 8: scrypt_core_kernelB_tex<8,2><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 9: scrypt_core_kernelB_tex<9,2><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 10: scrypt_core_kernelB_tex<10,2><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 11: scrypt_core_kernelB_tex<11,2><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 12: scrypt_core_kernelB_tex<12,2><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 13: scrypt_core_kernelB_tex<13,2><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 14: scrypt_core_kernelB_tex<14,2><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 15: scrypt_core_kernelB_tex<15,2><<< grid, threads, 0, stream >>>(d_odata); break;
                    case 16: scrypt_core_kernelB_tex<16,2><<< grid, threads, 0, stream >>>(d_odata); break;
                default: success = false; break;
            }
        } else success = false;
    }
    else
    {
        switch (WARPS_PER_BLOCK) {
            case 1: scrypt_core_kernelB<1><<< grid, threads, 0, stream >>>(d_odata); break;
            case 2: scrypt_core_kernelB<2><<< grid, threads, 0, stream >>>(d_odata); break;
            case 3: scrypt_core_kernelB<3><<< grid, threads, 0, stream >>>(d_odata); break;
            case 4: scrypt_core_kernelB<4><<< grid, threads, 0, stream >>>(d_odata); break;
            case 5: scrypt_core_kernelB<5><<< grid, threads, 0, stream >>>(d_odata); break;
            case 6: scrypt_core_kernelB<6><<< grid, threads, 0, stream >>>(d_odata); break;
            case 7: scrypt_core_kernelB<7><<< grid, threads, 0, stream >>>(d_odata); break;
            case 8: scrypt_core_kernelB<8><<< grid, threads, 0, stream >>>(d_odata); break;
            case 9: scrypt_core_kernelB<9><<< grid, threads, 0, stream >>>(d_odata); break;
            case 10: scrypt_core_kernelB<10><<< grid, threads, 0, stream >>>(d_odata); break;
            case 11: scrypt_core_kernelB<11><<< grid, threads, 0, stream >>>(d_odata); break;
            case 12: scrypt_core_kernelB<12><<< grid, threads, 0, stream >>>(d_odata); break;
            case 13: scrypt_core_kernelB<13><<< grid, threads, 0, stream >>>(d_odata); break;
            case 14: scrypt_core_kernelB<14><<< grid, threads, 0, stream >>>(d_odata); break;
            case 15: scrypt_core_kernelB<15><<< grid, threads, 0, stream >>>(d_odata); break;
            case 16: scrypt_core_kernelB<16><<< grid, threads, 0, stream >>>(d_odata); break;
            default: success = false; break;
        }
    }

    // catch any kernel launch failures
    if (cudaPeekAtLastError() != cudaSuccess) success = false;

    return success;
}

#define ROTL7(a0,a1,a2,a3,a00,a10,a20,a30){\
a0^=(((a00)<<7) | ((a00)>>25) );\
a1^=(((a10)<<7) | ((a10)>>25) );\
a2^=(((a20)<<7) | ((a20)>>25) );\
a3^=(((a30)<<7) | ((a30)>>25) );\
};\

#define ROTL9(a0,a1,a2,a3,a00,a10,a20,a30){\
a0^=(((a00)<<9) | ((a00)>>23) );\
a1^=(((a10)<<9) | ((a10)>>23) );\
a2^=(((a20)<<9) | ((a20)>>23) );\
a3^=(((a30)<<9) | ((a30)>>23) );\
};\

#define ROTL13(a0,a1,a2,a3,a00,a10,a20,a30){\
a0^=(((a00)<<13) | ((a00)>>19) );\
a1^=(((a10)<<13) | ((a10)>>19) );\
a2^=(((a20)<<13) | ((a20)>>19) );\
a3^=(((a30)<<13) | ((a30)>>19) );\
};\

#define ROTL18(a0,a1,a2,a3,a00,a10,a20,a30){\
a0^=(((a00)<<18) | ((a00)>>14) );\
a1^=(((a10)<<18) | ((a10)>>14) );\
a2^=(((a20)<<18) | ((a20)>>14) );\
a3^=(((a30)<<18) | ((a30)>>14) );\
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

static __device__ uint4& operator^=(uint4& left, const uint4& right)
{
    left.x ^= right.x;
    left.y ^= right.y;
    left.z ^= right.z;
    left.w ^= right.w;
    return left;
}

////////////////////////////////////////////////////////////////////////////////
//! Scrypt core kernel
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
template <int WARPS_PER_BLOCK> __global__ void
scrypt_core_kernelA(uint32_t *g_idata)
{
    __shared__ uint32_t X[WARPS_PER_BLOCK][WU_PER_WARP][16+1+_64BIT_ALIGN]; // +1 to resolve bank conflicts

    volatile int warpIdx        = threadIdx.x / warpSize;
    volatile int warpThread     = threadIdx.x % warpSize;

    // add block specific offsets
    int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_idata += 32 * offset;
    uint32_t * V = c_V[offset/WU_PER_WARP];

    // variables supporting the large memory transaction magic
    unsigned int Y = warpThread/4;
    unsigned int Z = 4*(warpThread%4);

    // registers to store an entire work unit
    uint32_t B[16], C[16];

    uint32_t ((*XB)[16+1+_64BIT_ALIGN]) = (uint32_t (*)[16+1+_64BIT_ALIGN])&X[warpIdx][Y][Z];
    uint32_t *XX = X[warpIdx][warpThread];

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&V[SCRATCH*(wu+Y)+Z])) = *((uint4*)(XB[wu])) = *((uint4*)(&g_idata[32*(wu+Y)+Z]));
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) B[idx] = XX[idx];

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&V[SCRATCH*(wu+Y)+16+Z])) = *((uint4*)(XB[wu])) = *((uint4*)(&g_idata[32*(wu+Y)+16+Z]));
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) C[idx] = XX[idx];

    for (int i = 1; i < 1024; i++) {

        xor_salsa8(B, C); xor_salsa8(C, B);

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) XX[idx] = B[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)(&V[SCRATCH*(wu+Y) + i*32 + Z])) = *((uint4*)(XB[wu]));

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) XX[idx] = C[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)(&V[SCRATCH*(wu+Y) + i*32 + 16 + Z])) = *((uint4*)(XB[wu]));
    }
}

template <int WARPS_PER_BLOCK> __global__ void
scrypt_core_kernelB(uint32_t *g_odata)
{
    __shared__ uint32_t X[WARPS_PER_BLOCK][WU_PER_WARP][16+1+_64BIT_ALIGN]; // +1 to resolve bank conflicts

    int warpIdx        = threadIdx.x / warpSize;
    int warpThread     = threadIdx.x % warpSize;

    // add block specific offsets
    int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_odata += 32 * offset;
    uint32_t * V = c_V[offset/WU_PER_WARP];

    // variables supporting the large memory transaction magic
    unsigned int Y = warpThread/4;
    unsigned int Z = 4*(warpThread%4);

    // registers to store an entire work unit
    uint32_t B[16], C[16];

    uint32_t ((*XB)[16+1+_64BIT_ALIGN]) = (uint32_t (*)[16+1+_64BIT_ALIGN])&X[warpIdx][Y][Z];
    uint32_t *XX = X[warpIdx][warpThread];

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(XB[wu])) = *((uint4*)(&V[SCRATCH*(wu+Y) + 1023*32 + Z]));
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) B[idx] = XX[idx];

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(XB[wu])) = *((uint4*)(&V[SCRATCH*(wu+Y) + 1023*32 + 16+Z]));
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) C[idx] = XX[idx];

    xor_salsa8(B, C); xor_salsa8(C, B);

    for (int i = 0; i < 1024; i++) {

        XX[16] = 32 * (C[0] & 1023);

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) XX[idx] = B[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)(XB[wu])) ^= *((uint4*)(&V[SCRATCH*(wu+Y) + X[warpIdx][wu+Y][16] + Z]));
#pragma unroll 16
        for (int idx=0; idx < 16; idx++) B[idx] = XX[idx];

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) XX[idx] = C[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)(XB[wu])) ^= *((uint4*)(&V[SCRATCH*(wu+Y) + X[warpIdx][wu+Y][16] + 16 + Z]));
#pragma unroll 16
        for (int idx=0; idx < 16; idx++) C[idx] = XX[idx];

        xor_salsa8(B, C); xor_salsa8(C, B);
    }

#pragma unroll 16
    for (int idx=0; idx < 16; ++idx) XX[idx] = B[idx];
#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&g_odata[32*(wu+Y)+Z])) = *((uint4*)(XB[wu]));

#pragma unroll 16
    for (int idx=0; idx < 16; ++idx) XX[idx] = C[idx];
#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&g_odata[32*(wu+Y)+16+Z])) = *((uint4*)(XB[wu]));
}

template <int WARPS_PER_BLOCK, int TEX_DIM> __global__ void
scrypt_core_kernelB_tex(uint32_t *g_odata)
{
    __shared__ uint32_t X[WARPS_PER_BLOCK][WU_PER_WARP][16+1+_64BIT_ALIGN]; // +1 to resolve bank conflicts

    volatile int warpIdx        = threadIdx.x / warpSize;
    volatile int warpThread     = threadIdx.x % warpSize;

    // add block specific offsets
    int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_odata += 32 * offset;

    // variables supporting the large memory transaction magic
    unsigned int Y = warpThread/4;
    unsigned int Z = 4*(warpThread%4);

    // registers to store an entire work unit
    uint32_t B[16], C[16];

    uint32_t ((*XB)[16+1+_64BIT_ALIGN]) = (uint32_t (*)[16+1+_64BIT_ALIGN])&X[warpIdx][Y][Z];
    uint32_t *XX = X[warpIdx][warpThread];

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(XB[wu])) = ((TEX_DIM == 1) ?
                    tex1Dfetch(texRef1D_4_V, (SCRATCH*(offset+wu+Y) + 1023*32 + Z)/4) :
                    tex2D(texRef2D_4_V, 0.5f + (32*1023 + Z)/4, 0.5f + (offset+wu+Y)));
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) B[idx] = XX[idx];

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(XB[wu])) = ((TEX_DIM == 1) ?
                    tex1Dfetch(texRef1D_4_V, (SCRATCH*(offset+wu+Y) + 1023*32 + 16+Z)/4) :
                    tex2D(texRef2D_4_V, 0.5f + (32*1023 + 16+Z)/4, 0.5f + (offset+wu+Y)));
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) C[idx] = XX[idx];

    xor_salsa8(B, C); xor_salsa8(C, B);

    for (int i = 0; i < 1024; i++) {

        XX[16] = 32 * (C[0] & 1023);

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) XX[idx] = B[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)(XB[wu])) ^= ((TEX_DIM == 1) ?
                        tex1Dfetch(texRef1D_4_V, (SCRATCH*(offset+wu+Y) + X[warpIdx][wu+Y][16] + Z)/4) :
                        tex2D(texRef2D_4_V, 0.5f + (X[warpIdx][wu+Y][16] + Z)/4, 0.5f + (offset+wu+Y)));
#pragma unroll 16
        for (int idx=0; idx < 16; idx++) B[idx] = XX[idx];

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) XX[idx] = C[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)(XB[wu])) ^= ((TEX_DIM == 1) ?
                        tex1Dfetch(texRef1D_4_V, (SCRATCH*(offset+wu+Y) + X[warpIdx][wu+Y][16] + 16+Z)/4) :
                        tex2D(texRef2D_4_V, 0.5f + (X[warpIdx][wu+Y][16] + 16+Z)/4, 0.5f + (offset+wu+Y)));
#pragma unroll 16
        for (int idx=0; idx < 16; idx++) C[idx] = XX[idx];

        xor_salsa8(B, C); xor_salsa8(C, B);
    }

#pragma unroll 16
    for (int idx=0; idx < 16; ++idx) XX[idx] = B[idx];
#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&g_odata[32*(wu+Y)+Z])) = *((uint4*)(XB[wu]));

#pragma unroll 16
    for (int idx=0; idx < 16; ++idx) XX[idx] = C[idx];
#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&g_odata[32*(wu+Y)+16+Z])) = *((uint4*)(XB[wu]));
}
