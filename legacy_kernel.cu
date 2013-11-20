//
// Kernel that runs best on Legacy (Compute 1.x) devices
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

#include "legacy_kernel.h"

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
template <int WARPS_PER_BLOCK> __global__ void legacy_scrypt_core_kernelA(uint32_t *g_idata);
template <int WARPS_PER_BLOCK> __global__ void legacy_scrypt_core_kernelB(uint32_t *g_odata);
template <int WARPS_PER_BLOCK, int TEX_DIM> __global__ void legacy_scrypt_core_kernelB_tex(uint32_t *g_odata);

// scratchbuf constants (pointers to scratch buffer for each work unit)
__constant__ uint32_t* c_V[1024];

// using texture references for the "tex" variants of the B kernels
texture<uint2, 1, cudaReadModeElementType> texRef1D_2_V;
texture<uint2, 2, cudaReadModeElementType> texRef2D_2_V;

LegacyKernel::LegacyKernel() : KernelInterface()
{
}

bool LegacyKernel::bindtexture_1D(uint32_t *d_V, size_t size)
{
    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<uint2>();
    texRef1D_2_V.normalized = 0;
    texRef1D_2_V.filterMode = cudaFilterModePoint;
    texRef1D_2_V.addressMode[0] = cudaAddressModeClamp;
    checkCudaErrors(cudaBindTexture(NULL, &texRef1D_2_V, d_V, &channelDesc2, size));
    return true;
}

bool LegacyKernel::bindtexture_2D(uint32_t *d_V, int width, int height, size_t pitch)
{
    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<uint2>();
    texRef2D_2_V.normalized = 0;
    texRef2D_2_V.filterMode = cudaFilterModePoint;
    texRef2D_2_V.addressMode[0] = cudaAddressModeClamp;
    texRef2D_2_V.addressMode[1] = cudaAddressModeClamp;
    checkCudaErrors(cudaBindTexture2D(NULL, &texRef2D_2_V, d_V, &channelDesc2, width, height, pitch));
    return true;
}

bool LegacyKernel::unbindtexture_1D()
{
    checkCudaErrors(cudaUnbindTexture(texRef1D_2_V));
    return true;
}

bool LegacyKernel::unbindtexture_2D()
{
    checkCudaErrors(cudaUnbindTexture(texRef2D_2_V));
    return true;
}

void LegacyKernel::set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_V, h_V, MAXWARPS*sizeof(uint32_t*), 0, cudaMemcpyHostToDevice));
}

bool LegacyKernel::run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, int *mutex, bool interactive, bool benchmark, int texture_cache)
{
    bool success = true;

    // clear CUDA's error variable
    cudaGetLastError();

    // First phase: Sequential writes to scratchpad.

    switch (WARPS_PER_BLOCK) {
        case 1: legacy_scrypt_core_kernelA<1><<< grid, threads, 0, stream >>>(d_idata); break;
        case 2: legacy_scrypt_core_kernelA<2><<< grid, threads, 0, stream >>>(d_idata); break;
        case 3: legacy_scrypt_core_kernelA<3><<< grid, threads, 0, stream >>>(d_idata); break;
//            case 4: legacy_scrypt_core_kernelA<4><<< grid, threads, 0, stream >>>(d_idata); break;
//            case 5: legacy_scrypt_core_kernelA<5><<< grid, threads, 0, stream >>>(d_idata); break;
//            case 6: legacy_scrypt_core_kernelA<6><<< grid, threads, 0, stream >>>(d_idata); break;
//            case 7: legacy_scrypt_core_kernelA<7><<< grid, threads, 0, stream >>>(d_idata); break;
//            case 8: legacy_scrypt_core_kernelA<8><<< grid, threads, 0, stream >>>(d_idata); break;
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
                case 1: legacy_scrypt_core_kernelB_tex<1,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 2: legacy_scrypt_core_kernelB_tex<2,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 3: legacy_scrypt_core_kernelB_tex<3,1><<< grid, threads, 0, stream >>>(d_odata); break;
//                    case 4: legacy_scrypt_core_kernelB_tex<4,1><<< grid, threads, 0, stream >>>(d_odata); break;
//                    case 5: legacy_scrypt_core_kernelB_tex<5,1><<< grid, threads, 0, stream >>>(d_odata); break;
//                    case 6: legacy_scrypt_core_kernelB_tex<6,1><<< grid, threads, 0, stream >>>(d_odata); break;
//                    case 7: legacy_scrypt_core_kernelB_tex<7,1><<< grid, threads, 0, stream >>>(d_odata); break;
//                    case 8: legacy_scrypt_core_kernelB_tex<8,1><<< grid, threads, 0, stream >>>(d_odata); break;
                default: success = false; break;
            }
        }
        else if (texture_cache == 2)
        {
            switch (WARPS_PER_BLOCK) {
                case 1: legacy_scrypt_core_kernelB_tex<1,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 2: legacy_scrypt_core_kernelB_tex<2,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 3: legacy_scrypt_core_kernelB_tex<3,2><<< grid, threads, 0, stream >>>(d_odata); break;
//                   case 4: legacy_scrypt_core_kernelB_tex<4,2><<< grid, threads, 0, stream >>>(d_odata); break;
//                   case 5: legacy_scrypt_core_kernelB_tex<5,2><<< grid, threads, 0, stream >>>(d_odata); break;
//                   case 6: legacy_scrypt_core_kernelB_tex<6,2><<< grid, threads, 0, stream >>>(d_odata); break;
//                   case 7: legacy_scrypt_core_kernelB_tex<7,2><<< grid, threads, 0, stream >>>(d_odata); break;
//                   case 8: legacy_scrypt_core_kernelB_tex<8,2><<< grid, threads, 0, stream >>>(d_odata); break;
                default: success = false; break;
            }
        } else success = false;
    }
    else
    {
        switch (WARPS_PER_BLOCK) {
            case 1: legacy_scrypt_core_kernelB<1><<< grid, threads, 0, stream >>>(d_odata); break;
            case 2: legacy_scrypt_core_kernelB<2><<< grid, threads, 0, stream >>>(d_odata); break;
            case 3: legacy_scrypt_core_kernelB<3><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 4: legacy_scrypt_core_kernelB<4><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 5: legacy_scrypt_core_kernelB<5><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 6: legacy_scrypt_core_kernelB<6><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 7: legacy_scrypt_core_kernelB<7><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 8: legacy_scrypt_core_kernelB<8><<< grid, threads, 0, stream >>>(d_odata); break;
            default: success = false; break;
        }
    }

    // catch any kernel launch failures
    if (cudaPeekAtLastError() != cudaSuccess) success = false;

    return success;
}

#define ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))

static __host__ __device__ void xor_salsa8(uint32_t * const B, const uint32_t * const C)
{
    // the "volatile" puts data into registers right away
    volatile uint32_t x0 = (B[ 0] ^= C[ 0]), x1 = (B[ 1] ^= C[ 1]), x2 = (B[ 2] ^= C[ 2]), x3 = (B[ 3] ^= C[ 3]);
    volatile uint32_t x4 = (B[ 4] ^= C[ 4]), x5 = (B[ 5] ^= C[ 5]), x6 = (B[ 6] ^= C[ 6]), x7 = (B[ 7] ^= C[ 7]);
    volatile uint32_t x8 = (B[ 8] ^= C[ 8]), x9 = (B[ 9] ^= C[ 9]), xa = (B[10] ^= C[10]), xb = (B[11] ^= C[11]);
    volatile uint32_t xc = (B[12] ^= C[12]), xd = (B[13] ^= C[13]), xe = (B[14] ^= C[14]), xf = (B[15] ^= C[15]);

    for (int i = 0; i < 4; ++i)
    {
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

static __host__ __device__ uint2& operator^=(uint2& left, const uint2& right)
{
    left.x ^= right.x;
    left.y ^= right.y;
    return left;
}

////////////////////////////////////////////////////////////////////////////////
//! Scrypt core kernel with higher shared memory use (faster on older devices)
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
template <int WARPS_PER_BLOCK> __global__ void
legacy_scrypt_core_kernelA(uint32_t *g_idata)
{
    __shared__ uint32_t X[WARPS_PER_BLOCK][WU_PER_WARP][32+1+_64BIT_ALIGN]; // +1 to resolve bank conflicts

    volatile int warpIdx    = threadIdx.x / warpSize;
    volatile int warpThread = threadIdx.x % warpSize;

    // variables supporting the large memory transaction magic
    volatile unsigned int Y = warpThread/16;
    volatile unsigned int Z = 2*(warpThread%16);

    // add block specific offsets
    volatile int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_idata += 32      * offset;
    uint32_t * volatile V = c_V[offset / WU_PER_WARP] + SCRATCH*Y + Z;

    uint32_t ((*XB)[32+1+_64BIT_ALIGN]) = (uint32_t (*)[32+1+_64BIT_ALIGN])&X[warpIdx][Y][Z];
    uint32_t *XX = X[warpIdx][warpThread];

    {
#pragma unroll 16
        for (int wu=0; wu < 32; wu+=2)
            *((uint2*)XB[wu]) = *((uint2*)(&g_idata[32*(wu+Y)+Z]));

#pragma unroll 16
        for (int wu=0; wu < 32; wu+=2)
            *((uint2*)(&V[SCRATCH*wu])) = *((uint2*)XB[wu]);

        for (int i = 1; i < 1024; i++)
        {
            xor_salsa8(&XX[0], &XX[16]);
            xor_salsa8(&XX[16], &XX[0]);

#pragma unroll 16
            for (int wu=0; wu < 32; wu+=2)
                *((uint2*)(&V[SCRATCH*wu + i*32])) = *((uint2*)XB[wu]);
        }
    }
}

template <int WARPS_PER_BLOCK> __global__ void
legacy_scrypt_core_kernelB(uint32_t *g_odata)
{
    __shared__ uint32_t X[WARPS_PER_BLOCK][WU_PER_WARP][32+1+_64BIT_ALIGN]; // +1 to resolve bank conflicts

    volatile int warpIdx    = threadIdx.x / warpSize;
    volatile int warpThread = threadIdx.x % warpSize;

    // variables supporting the large memory transaction magic
    volatile unsigned int Y = warpThread/16;
    volatile unsigned int Z = 2*(warpThread%16);

    // add block specific offsets
    volatile int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_odata += 32      * offset;
    uint32_t * volatile V = c_V[offset / WU_PER_WARP] + SCRATCH*Y + Z;

    uint32_t ((*XB)[32+1+_64BIT_ALIGN]) = (uint32_t (*)[32+1+_64BIT_ALIGN])&X[warpIdx][Y][Z];
    uint32_t *XX = X[warpIdx][warpThread];

    {
#pragma unroll 16
        for (int wu=0; wu < 32; wu+=2)
            *((uint2*)XB[wu]) = *((uint2*)(&V[SCRATCH*wu + 1023*32]));

        xor_salsa8(&XX[0], &XX[16]);
        xor_salsa8(&XX[16], &XX[0]);

        for (int i = 0; i < 1024; i++)
        {
#pragma unroll 16
        for (int wu=0; wu < 32; wu+=2)
            *((uint2*)XB[wu]) ^= *((uint2*)(&V[SCRATCH*wu + 32*(X[warpIdx][wu+Y][16] & 1023)]));

            xor_salsa8(&XX[0], &XX[16]);
            xor_salsa8(&XX[16], &XX[0]);
        }

#pragma unroll 16
        for (int wu=0; wu < 32; wu+=2)
            *((uint2*)(&g_odata[32*(wu+Y)+Z])) = *((uint2*)XB[wu]);
    }
}

template <int WARPS_PER_BLOCK, int TEX_DIM> __global__ void
legacy_scrypt_core_kernelB_tex(uint32_t *g_odata)
{
    __shared__ uint32_t X[WARPS_PER_BLOCK][WU_PER_WARP][32+1+_64BIT_ALIGN]; // +1 to resolve bank conflicts

    volatile int warpIdx    = threadIdx.x / warpSize;
    volatile int warpThread = threadIdx.x % warpSize;

    // variables supporting the large memory transaction magic
    volatile unsigned int Y = warpThread/16;
    volatile unsigned int Z = 2*(warpThread%16);

    // add block specific offsets
    volatile int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_odata += 32      * offset;

    uint32_t ((*XB)[32+1+_64BIT_ALIGN]) = (uint32_t (*)[32+1+_64BIT_ALIGN])&X[warpIdx][Y][Z];
    uint32_t *XX = X[warpIdx][warpThread];

    {
#pragma unroll 16
        for (int wu=0; wu < 32; wu+=2)
            *((uint2*)XB[wu]) = ((TEX_DIM == 1) ?
                        tex1Dfetch(texRef1D_2_V, (SCRATCH*(offset+wu+Y) + 1023*32 + Z)/2) :
                        tex2D(texRef2D_2_V, 0.5f + (32*1023 + Z)/2, 0.5f + (offset+wu+Y)));

        xor_salsa8(&XX[0], &XX[16]);
        xor_salsa8(&XX[16], &XX[0]);

        for (int i = 0; i < 1024; i++)
        {
#pragma unroll 16
        for (int wu=0; wu < 32; wu+=2)
            *((uint2*)XB[wu]) ^= ((TEX_DIM == 1) ?
                        tex1Dfetch(texRef1D_2_V, (SCRATCH*(offset+wu+Y) + 32*(X[warpIdx][wu+Y][16] & 1023) + Z)/2) :
                        tex2D(texRef2D_2_V, 0.5f + (32*(X[warpIdx][wu+Y][16] & 1023) + Z)/2, 0.5f + (offset+wu+Y)));

            xor_salsa8(&XX[0], &XX[16]);
            xor_salsa8(&XX[16], &XX[0]);
        }

#pragma unroll 16
        for (int wu=0; wu < 32; wu+=2)
            *((uint2*)(&g_odata[32*(wu+Y)+Z])) = *((uint2*)XB[wu]);
    }
}

