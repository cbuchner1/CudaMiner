//
// Kernel that runs best on Kepler (Compute 3.0) devices
//
// - makes use of 8 byte of Kepler's shared memory bank mode
// - does memory transfers with ulonglong2 vectors whereever possible
// - further halves shared memory consumption over Fermi kernel by sharing
//   the same shared memory buffers among two neighbor warps. Uses spinlocks
//   based on shared memory atomics and Compute 3.0
//  
// NOTE: compile this .cu module for compute_30,sm_30 with --maxrregcount=63
//       but for compatibility with CUDA 5.5 tookit, we use compute_12,sm_12
//       and we limit the maximum. no. of warps to 12
//

#ifdef WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include <cuda.h>

#include "spinlock_kernel.h"

// forward references
template <int WARPS_PER_BLOCK> __global__ void spinlock_scrypt_core_kernelA(uint32_t *g_idata);
template <int WARPS_PER_BLOCK> __global__ void spinlock_scrypt_core_kernelB(uint32_t *g_odata);
template <int WARPS_PER_BLOCK, int TEX_DIM> __global__ void spinlock_scrypt_core_kernelB_tex(uint32_t *g_odata);

// scratchbuf constants (pointers to scratch buffer for each work unit)
__constant__ uint32_t* c_V[1024];

// using texture references for the "tex" variants of the B kernels
texture<uint4, 1, cudaReadModeElementType> texRef1D_4_V;
texture<uint4, 2, cudaReadModeElementType> texRef2D_4_V;

SpinlockKernel::SpinlockKernel() : KernelInterface()
{
}

bool SpinlockKernel::bindtexture_1D(uint32_t *d_V, size_t size)
{
    cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc<uint4>();
    texRef1D_4_V.normalized = 0;
    texRef1D_4_V.filterMode = cudaFilterModePoint;
    texRef1D_4_V.addressMode[0] = cudaAddressModeClamp;
    checkCudaErrors(cudaBindTexture(NULL, &texRef1D_4_V, d_V, &channelDesc4, size));
    return true;
}

bool SpinlockKernel::bindtexture_2D(uint32_t *d_V, int width, int height, size_t pitch)
{
    cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc<uint4>();
    texRef2D_4_V.normalized = 0;
    texRef2D_4_V.filterMode = cudaFilterModePoint;
    texRef2D_4_V.addressMode[0] = cudaAddressModeClamp;
    texRef2D_4_V.addressMode[1] = cudaAddressModeClamp;
    checkCudaErrors(cudaBindTexture2D(NULL, &texRef2D_4_V, d_V, &channelDesc4, width, height, pitch));
    return true;
}

bool SpinlockKernel::unbindtexture_1D()
{
    checkCudaErrors(cudaUnbindTexture(texRef1D_4_V));
    return true;
}

bool SpinlockKernel::unbindtexture_2D()
{
    checkCudaErrors(cudaUnbindTexture(texRef2D_4_V));
    return true;
}

void SpinlockKernel::set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_V, h_V, MAXWARPS*sizeof(uint32_t*), 0, cudaMemcpyHostToDevice));
}

bool SpinlockKernel::run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, bool interactive, bool benchmark, int texture_cache)
{
    bool success = true;

    // clear CUDA's error variable
    cudaGetLastError();

    // First phase: Sequential writes to scratchpad.

    switch (WARPS_PER_BLOCK) {
        case 1: spinlock_scrypt_core_kernelA<1><<< grid, threads, 0, stream >>>(d_idata); break;
        case 2: spinlock_scrypt_core_kernelA<2><<< grid, threads, 0, stream >>>(d_idata); break;
        case 3: spinlock_scrypt_core_kernelA<3><<< grid, threads, 0, stream >>>(d_idata); break;
        case 4: spinlock_scrypt_core_kernelA<4><<< grid, threads, 0, stream >>>(d_idata); break;
        case 5: spinlock_scrypt_core_kernelA<5><<< grid, threads, 0, stream >>>(d_idata); break;
        case 6: spinlock_scrypt_core_kernelA<6><<< grid, threads, 0, stream >>>(d_idata); break;
        case 7: spinlock_scrypt_core_kernelA<7><<< grid, threads, 0, stream >>>(d_idata); break;
        case 8: spinlock_scrypt_core_kernelA<8><<< grid, threads, 0, stream >>>(d_idata); break;
        case 9: spinlock_scrypt_core_kernelA<9><<< grid, threads, 0, stream >>>(d_idata); break;
        case 10: spinlock_scrypt_core_kernelA<10><<< grid, threads, 0, stream >>>(d_idata); break;
        case 11: spinlock_scrypt_core_kernelA<11><<< grid, threads, 0, stream >>>(d_idata); break;
        case 12: spinlock_scrypt_core_kernelA<12><<< grid, threads, 0, stream >>>(d_idata); break;
//        case 13: spinlock_scrypt_core_kernelA<13><<< grid, threads, 0, stream >>>(d_idata); break;
//        case 14: spinlock_scrypt_core_kernelA<14><<< grid, threads, 0, stream >>>(d_idata); break;
//        case 15: spinlock_scrypt_core_kernelA<15><<< grid, threads, 0, stream >>>(d_idata); break;
//        case 16: spinlock_scrypt_core_kernelA<16><<< grid, threads, 0, stream >>>(d_idata); break;
//        case 17: spinlock_scrypt_core_kernelA<17><<< grid, threads, 0, stream >>>(d_idata); break;
//        case 18: spinlock_scrypt_core_kernelA<18><<< grid, threads, 0, stream >>>(d_idata); break;
//        case 19: spinlock_scrypt_core_kernelA<19><<< grid, threads, 0, stream >>>(d_idata); break;
//        case 20: spinlock_scrypt_core_kernelA<20><<< grid, threads, 0, stream >>>(d_idata); break;
//        case 21: spinlock_scrypt_core_kernelA<21><<< grid, threads, 0, stream >>>(d_idata); break;
//        case 22: spinlock_scrypt_core_kernelA<22><<< grid, threads, 0, stream >>>(d_idata); break;
//        case 23: spinlock_scrypt_core_kernelA<23><<< grid, threads, 0, stream >>>(d_idata); break;
//        case 24: spinlock_scrypt_core_kernelA<24><<< grid, threads, 0, stream >>>(d_idata); break;
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
                case 1: spinlock_scrypt_core_kernelB_tex<1,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 2: spinlock_scrypt_core_kernelB_tex<2,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 3: spinlock_scrypt_core_kernelB_tex<3,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 4: spinlock_scrypt_core_kernelB_tex<4,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 5: spinlock_scrypt_core_kernelB_tex<5,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 6: spinlock_scrypt_core_kernelB_tex<6,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 7: spinlock_scrypt_core_kernelB_tex<7,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 8: spinlock_scrypt_core_kernelB_tex<8,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 9: spinlock_scrypt_core_kernelB_tex<9,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 10: spinlock_scrypt_core_kernelB_tex<10,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 11: spinlock_scrypt_core_kernelB_tex<11,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 12: spinlock_scrypt_core_kernelB_tex<12,1><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 13: spinlock_scrypt_core_kernelB_tex<13,1><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 14: spinlock_scrypt_core_kernelB_tex<14,1><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 15: spinlock_scrypt_core_kernelB_tex<15,1><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 16: spinlock_scrypt_core_kernelB_tex<16,1><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 17: spinlock_scrypt_core_kernelB_tex<17,1><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 18: spinlock_scrypt_core_kernelB_tex<18,1><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 19: spinlock_scrypt_core_kernelB_tex<19,1><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 20: spinlock_scrypt_core_kernelB_tex<20,1><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 21: spinlock_scrypt_core_kernelB_tex<21,1><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 22: spinlock_scrypt_core_kernelB_tex<22,1><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 23: spinlock_scrypt_core_kernelB_tex<23,1><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 24: spinlock_scrypt_core_kernelB_tex<24,1><<< grid, threads, 0, stream >>>(d_odata); break;
                default: success = false; break;
            }
        }
        else if (texture_cache == 2)
        {
            switch (WARPS_PER_BLOCK) {
                case 1: spinlock_scrypt_core_kernelB_tex<1,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 2: spinlock_scrypt_core_kernelB_tex<2,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 3: spinlock_scrypt_core_kernelB_tex<3,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 4: spinlock_scrypt_core_kernelB_tex<4,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 5: spinlock_scrypt_core_kernelB_tex<5,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 6: spinlock_scrypt_core_kernelB_tex<6,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 7: spinlock_scrypt_core_kernelB_tex<7,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 8: spinlock_scrypt_core_kernelB_tex<8,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 9: spinlock_scrypt_core_kernelB_tex<9,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 10: spinlock_scrypt_core_kernelB_tex<10,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 11: spinlock_scrypt_core_kernelB_tex<11,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 12: spinlock_scrypt_core_kernelB_tex<12,2><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 13: spinlock_scrypt_core_kernelB_tex<13,2><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 14: spinlock_scrypt_core_kernelB_tex<14,2><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 15: spinlock_scrypt_core_kernelB_tex<15,2><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 16: spinlock_scrypt_core_kernelB_tex<16,2><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 17: spinlock_scrypt_core_kernelB_tex<17,2><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 18: spinlock_scrypt_core_kernelB_tex<18,2><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 19: spinlock_scrypt_core_kernelB_tex<19,2><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 20: spinlock_scrypt_core_kernelB_tex<20,2><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 21: spinlock_scrypt_core_kernelB_tex<21,2><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 22: spinlock_scrypt_core_kernelB_tex<22,2><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 23: spinlock_scrypt_core_kernelB_tex<23,2><<< grid, threads, 0, stream >>>(d_odata); break;
//                case 24: spinlock_scrypt_core_kernelB_tex<24,2><<< grid, threads, 0, stream >>>(d_odata); break;
                default: success = false; break;
            }
        } else success = false;
    }
    else
    {
        switch (WARPS_PER_BLOCK) {
            case 1: spinlock_scrypt_core_kernelB<1><<< grid, threads, 0, stream >>>(d_odata); break;
            case 2: spinlock_scrypt_core_kernelB<2><<< grid, threads, 0, stream >>>(d_odata); break;
            case 3: spinlock_scrypt_core_kernelB<3><<< grid, threads, 0, stream >>>(d_odata); break;
            case 4: spinlock_scrypt_core_kernelB<4><<< grid, threads, 0, stream >>>(d_odata); break;
            case 5: spinlock_scrypt_core_kernelB<5><<< grid, threads, 0, stream >>>(d_odata); break;
            case 6: spinlock_scrypt_core_kernelB<6><<< grid, threads, 0, stream >>>(d_odata); break;
            case 7: spinlock_scrypt_core_kernelB<7><<< grid, threads, 0, stream >>>(d_odata); break;
            case 8: spinlock_scrypt_core_kernelB<8><<< grid, threads, 0, stream >>>(d_odata); break;
            case 9: spinlock_scrypt_core_kernelB<9><<< grid, threads, 0, stream >>>(d_odata); break;
            case 10: spinlock_scrypt_core_kernelB<10><<< grid, threads, 0, stream >>>(d_odata); break;
            case 11: spinlock_scrypt_core_kernelB<11><<< grid, threads, 0, stream >>>(d_odata); break;
            case 12: spinlock_scrypt_core_kernelB<12><<< grid, threads, 0, stream >>>(d_odata); break;
//            case 13: spinlock_scrypt_core_kernelB<13><<< grid, threads, 0, stream >>>(d_odata); break;
//            case 14: spinlock_scrypt_core_kernelB<14><<< grid, threads, 0, stream >>>(d_odata); break;
//            case 15: spinlock_scrypt_core_kernelB<15><<< grid, threads, 0, stream >>>(d_odata); break;
//            case 16: spinlock_scrypt_core_kernelB<16><<< grid, threads, 0, stream >>>(d_odata); break;
//            case 17: spinlock_scrypt_core_kernelB<17><<< grid, threads, 0, stream >>>(d_odata); break;
//            case 18: spinlock_scrypt_core_kernelB<18><<< grid, threads, 0, stream >>>(d_odata); break;
//            case 19: spinlock_scrypt_core_kernelB<19><<< grid, threads, 0, stream >>>(d_odata); break;
//            case 20: spinlock_scrypt_core_kernelB<20><<< grid, threads, 0, stream >>>(d_odata); break;
//            case 21: spinlock_scrypt_core_kernelB<21><<< grid, threads, 0, stream >>>(d_odata); break;
//            case 22: spinlock_scrypt_core_kernelB<22><<< grid, threads, 0, stream >>>(d_odata); break;
//            case 23: spinlock_scrypt_core_kernelB<23><<< grid, threads, 0, stream >>>(d_odata); break;
//            case 24: spinlock_scrypt_core_kernelB<24><<< grid, threads, 0, stream >>>(d_odata); break;
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

static __device__ void lock(int *mutex, int i)
{
    while( atomicCAS( &mutex[i], 0, 1 ) != 0 );
}

static __device__ void unlock(int *mutex, int i)
{
    atomicExch( &mutex[i], 0 );
}

////////////////////////////////////////////////////////////////////////////////
//! Scrypt core kernel using spinlocks to cut shared memory use in half.
//! Ideal for Kepler devices where shared memory use prevented optimal occupancy.
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
template <int WARPS_PER_BLOCK> __global__ void
spinlock_scrypt_core_kernelA(uint32_t *g_idata)
{
    __shared__ uint32_t X[(WARPS_PER_BLOCK+1)/2][WU_PER_WARP][16+4];
    __shared__ int      L[(WARPS_PER_BLOCK+1)/2];

    int warpIdx        = threadIdx.x / warpSize;
    int warpThread     = threadIdx.x % warpSize;

    // variables supporting the large memory transaction magic
    unsigned int Y = warpThread/4;
    unsigned int Z = 4*(warpThread%4);

    // add block specific offsets
    int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_idata += 32 * offset;
    uint32_t * V = c_V[offset / WU_PER_WARP]  + SCRATCH*Y + Z;

    // registers to store an entire work unit
    uint4 B[4], C[4];

    volatile int warpIdx_2      = warpIdx/2;
    uint32_t ((*XB)[16+4]) = (uint32_t (*)[16+4])&X[warpIdx_2][Y][Z];
    uint32_t *XX = X[warpIdx_2][warpThread];

    if (threadIdx.x < (WARPS_PER_BLOCK+1)/2) L[threadIdx.x] = 0;
    __syncthreads();
    if (warpThread == 0) lock(L, warpIdx_2);

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

        if (warpThread == 0) unlock(L, warpIdx_2);
        xor_salsa8(B, C); xor_salsa8(C, B);
        if (warpThread == 0) lock(L, warpIdx_2);

#pragma unroll 4
        for (int idx=0; idx < 4; idx++) *((uint4*)&XX[4*idx]) = B[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((ulonglong2*)(&V[SCRATCH*wu + i*32])) = *((ulonglong2*)XB[wu]);

#pragma unroll 4
        for (int idx=0; idx < 4; idx++) *((uint4*)&XX[4*idx]) = C[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((ulonglong2*)(&V[SCRATCH*wu + i*32 + 16])) = *((ulonglong2*)XB[wu]);
    }
    if (warpThread == 0) unlock(L, warpIdx_2);
}

template <int WARPS_PER_BLOCK> __global__ void
spinlock_scrypt_core_kernelB(uint32_t *g_odata)
{
    __shared__ uint32_t X[(WARPS_PER_BLOCK+1)/2][WU_PER_WARP][16+4];
    __shared__ int      L[(WARPS_PER_BLOCK+1)/2];

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

    volatile int warpIdx_2      = warpIdx/2;
    uint32_t ((*XB)[16+4]) = (uint32_t (*)[16+4])&X[warpIdx_2][Y][Z];
    uint32_t *XX = X[warpIdx_2][warpThread];

    if (threadIdx.x < (WARPS_PER_BLOCK+1)/2) L[threadIdx.x] = 0;
    __syncthreads();
    if (warpThread == 0) lock(L, warpIdx_2);

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((ulonglong2*)XB[wu]) = *((ulonglong2*)(&V[SCRATCH*wu + 1023*32]));
#pragma unroll 4
    for (int idx=0; idx < 4; idx++) B[idx] = *((uint4*)&XX[4*idx]);

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((ulonglong2*)XB[wu]) = *((ulonglong2*)(&V[SCRATCH*wu + 1023*32 + 16]));
#pragma unroll 4
    for (int idx=0; idx < 4; idx++) C[idx] = *((uint4*)&XX[4*idx]);

    if (warpThread == 0) unlock(L, warpIdx_2);
    xor_salsa8(B, C); xor_salsa8(C, B);
    if (warpThread == 0) lock(L, warpIdx_2);

    for (int i = 0; i < 1024; i++) {

        XX[16] = 32 * (C[0].x & 1023);

#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((ulonglong2*)XB[wu]) = *((ulonglong2*)(&V[SCRATCH*wu + XB[wu][16-Z]]));
#pragma unroll 4
        for (int idx=0; idx < 4; idx++) B[idx] ^= *((uint4*)&XX[4*idx]);

#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((ulonglong2*)XB[wu]) = *((ulonglong2*)(&V[SCRATCH*wu + XB[wu][16-Z] + 16]));
#pragma unroll 4
        for (int idx=0; idx < 4; idx++) C[idx] ^= *((uint4*)&XX[4*idx]);

        if (warpThread == 0) unlock(L, warpIdx_2);
        xor_salsa8(B, C); xor_salsa8(C, B);
        if (warpThread == 0) lock(L, warpIdx_2);
    }

#pragma unroll 4
    for (int idx=0; idx < 4; idx++) *((uint4*)&XX[4*idx]) = B[idx];
#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((ulonglong2*)(&g_odata[32*(wu+Y)+Z])) = *((ulonglong2*)XB[wu]);

#pragma unroll 4
    for (int idx=0; idx < 4; idx++) *((uint4*)&XX[4*idx]) = C[idx];
#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((ulonglong2*)(&g_odata[32*(wu+Y)+16+Z])) = *((ulonglong2*)XB[wu]);

    if (warpThread == 0) unlock(L, warpIdx_2);
}

template <int WARPS_PER_BLOCK, int TEX_DIM> __global__ void
spinlock_scrypt_core_kernelB_tex(uint32_t *g_odata)
{
    __shared__ uint32_t X[(WARPS_PER_BLOCK+1)/2][WU_PER_WARP][16+4];
    __shared__ int      L[(WARPS_PER_BLOCK+1)/2];

    int warpIdx        = threadIdx.x / warpSize;
    int warpThread     = threadIdx.x % warpSize;

    // variables supporting the large memory transaction magic
    unsigned int Y = warpThread/4;
    unsigned int Z = 4*(warpThread%4);

    // add block specific offsets
    int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_odata += 32 * offset;

    // registers to store an entire work unit
    uint4 B[4], C[4];

    volatile int warpIdx_2      = warpIdx/2;
    uint32_t ((*XB)[16+4]) = (uint32_t (*)[16+4])&X[warpIdx_2][Y][Z];
    uint32_t *XX = X[warpIdx_2][warpThread];

    if (threadIdx.x < (WARPS_PER_BLOCK+1)/2) L[threadIdx.x] = 0;
    __syncthreads();
    if (warpThread == 0) lock(L, warpIdx_2);

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)XB[wu]) = ((TEX_DIM == 1) ?
                    tex1Dfetch(texRef1D_4_V, (SCRATCH*(offset+wu+Y) + 1023*32 + Z)/4) :
                    tex2D(texRef2D_4_V, 0.5f + (32*1023 + Z)/4, 0.5f + (offset+wu+Y)));
#pragma unroll 4
    for (int idx=0; idx < 4; idx++) B[idx] = *((uint4*)&XX[4*idx]);

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)XB[wu]) = ((TEX_DIM == 1) ?
                    tex1Dfetch(texRef1D_4_V, (SCRATCH*(offset+wu+Y) + 1023*32 + 16+Z)/4) :
                    tex2D(texRef2D_4_V, 0.5f + (32*1023 + 16+Z)/4, 0.5f + (offset+wu+Y)));
#pragma unroll 4
    for (int idx=0; idx < 4; idx++) C[idx] = *((uint4*)&XX[4*idx]);

    if (warpThread == 0) unlock(L, warpIdx_2);
    xor_salsa8(B, C); xor_salsa8(C, B);
    if (warpThread == 0) lock(L, warpIdx_2);

    for (int i = 0; i < 1024; i++) {

        XX[16] = 32 * (C[0].x & 1023);

#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)XB[wu]) = ((TEX_DIM == 1) ?
                        tex1Dfetch(texRef1D_4_V, (SCRATCH*(offset+wu+Y) + XB[wu][16-Z] + Z)/4) :
                        tex2D(texRef2D_4_V, 0.5f + (XB[wu][16-Z] + Z)/4, 0.5f + (offset+wu+Y)));
#pragma unroll 4
        for (int idx=0; idx < 4; idx++) B[idx] ^= *((uint4*)&XX[4*idx]);

#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)XB[wu]) = ((TEX_DIM == 1) ?
                        tex1Dfetch(texRef1D_4_V, (SCRATCH*(offset+wu+Y) + XB[wu][16-Z] + 16+Z)/4) :
                        tex2D(texRef2D_4_V, 0.5f + (XB[wu][16-Z] + 16+Z)/4, 0.5f + (offset+wu+Y)));
#pragma unroll 4
        for (int idx=0; idx < 4; idx++) C[idx] ^= *((uint4*)&XX[4*idx]);

        if (warpThread == 0) unlock(L, warpIdx_2);
        xor_salsa8(B, C); xor_salsa8(C, B);
        if (warpThread == 0) lock(L, warpIdx_2);
    }

#pragma unroll 4
    for (int idx=0; idx < 4; idx++) *((uint4*)&XX[4*idx]) = B[idx];
#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((ulonglong2*)(&g_odata[32*(wu+Y)+Z])) = *((ulonglong2*)XB[wu]);

#pragma unroll 4
    for (int idx=0; idx < 4; idx++) *((uint4*)&XX[4*idx]) = C[idx];
#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((ulonglong2*)(&g_odata[32*(wu+Y)+16+Z])) = *((ulonglong2*)XB[wu]);

    if (warpThread == 0) unlock(L, warpIdx_2);
}
