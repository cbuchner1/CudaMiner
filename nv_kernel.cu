//
// Experimental Kernel for Kepler (Compute 3.5) devices
// code submitted by nVidia performance engineer Alexey Panteleev
// with modifications by Christian Buchner
//
// for Compute 3.5
// NOTE: compile this .cu module for compute_35,sm_35 with --maxrregcount=80
// for Compute 3.0
// NOTE: compile this .cu module for compute_30,sm_30 with --maxrregcount=63
//

#include <map>

#ifdef WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include <cuda.h>

#include "miner.h"
#include "nv_kernel.h"

#define THREADS_PER_WU 1  // single thread per hash 

#define TEXWIDTH 32768

#if __CUDA_ARCH__ < 350 
    // Kepler (Compute 3.0)
    #define __ldg(x) (*(x))
#endif

// grab lane ID
static __device__ __inline__ unsigned int __laneId() { unsigned int laneId; asm( "mov.u32 %0, %%laneid;" : "=r"( laneId ) ); return laneId; }

// forward references
template <int ALGO> __global__ void nv_scrypt_core_kernelA(uint32_t *g_idata, int begin, int end);
template <int ALGO, int TEX_DIM> __global__ void nv_scrypt_core_kernelB(uint32_t *g_odata, int begin, int end);
template <int ALGO> __global__ void nv_scrypt_core_kernelA_LG(uint32_t *g_idata, int begin, int end, unsigned int LOOKUP_GAP);
template <int ALGO, int TEX_DIM> __global__ void nv_scrypt_core_kernelB_LG(uint32_t *g_odata, int begin, int end, unsigned int LOOKUP_GAP);

// scratchbuf constants (pointers to scratch buffer for each work unit)
__constant__ uint32_t* c_V[TOTAL_WARP_LIMIT];

// using texture references for the "tex" variants of the B kernels
texture<uint4, 1, cudaReadModeElementType> texRef1D_4_V;
texture<uint4, 2, cudaReadModeElementType> texRef2D_4_V;

// iteration count N
__constant__ uint32_t c_N;
__constant__ uint32_t c_N_1; // N - 1
__constant__ uint32_t c_spacing; // (N+LOOKUP_GAP-1)/LOOKUP_GAP

NVKernel::NVKernel() : KernelInterface()
{
}

bool NVKernel::bindtexture_1D(uint32_t *d_V, size_t size)
{
    cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc<uint4>();
    texRef1D_4_V.normalized = 0;
    texRef1D_4_V.filterMode = cudaFilterModePoint;
    texRef1D_4_V.addressMode[0] = cudaAddressModeClamp;
    checkCudaErrors(cudaBindTexture(NULL, &texRef1D_4_V, d_V, &channelDesc4, size));
    return true;
}

bool NVKernel::bindtexture_2D(uint32_t *d_V, int width, int height, size_t pitch)
{
    cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc<uint4>();
    texRef2D_4_V.normalized = 0;
    texRef2D_4_V.filterMode = cudaFilterModePoint;
    texRef2D_4_V.addressMode[0] = cudaAddressModeClamp;
    texRef2D_4_V.addressMode[1] = cudaAddressModeClamp;
    // maintain texture width of TEXWIDTH (max. limit is 65000)
    while (width > TEXWIDTH) { width /= 2; height *= 2; pitch /= 2; }
    while (width < TEXWIDTH) { width *= 2; height = (height+1)/2; pitch *= 2; }
    checkCudaErrors(cudaBindTexture2D(NULL, &texRef2D_4_V, d_V, &channelDesc4, width, height, pitch));
    return true;
}

bool NVKernel::unbindtexture_1D()
{
    checkCudaErrors(cudaUnbindTexture(texRef1D_4_V));
    return true;
}

bool NVKernel::unbindtexture_2D()
{
    checkCudaErrors(cudaUnbindTexture(texRef2D_4_V));
    return true;
}

void NVKernel::set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_V, h_V, MAXWARPS*sizeof(uint32_t*), 0, cudaMemcpyHostToDevice));
}

bool NVKernel::run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, unsigned int N, unsigned int LOOKUP_GAP, bool interactive, bool benchmark, int texture_cache)
{
    bool success = true;

    // make some constants available to kernel, update only initially and when changing
    static int prev_N[MAX_DEVICES] = {0};
    if (N != prev_N[thr_id]) {
        uint32_t h_N = N;
        uint32_t h_N_1 = N-1;
        uint32_t h_spacing = (N+LOOKUP_GAP-1)/LOOKUP_GAP;

        cudaMemcpyToSymbolAsync(c_N, &h_N, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream);
        cudaMemcpyToSymbolAsync(c_N_1, &h_N_1, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream);
        cudaMemcpyToSymbolAsync(c_spacing, &h_spacing, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream);

        prev_N[thr_id] = N;
    }

    // First phase: Sequential writes to scratchpad.
    const int batch = device_batchsize[thr_id];
    unsigned int pos = 0;
    
    do 
    {
        if (LOOKUP_GAP == 1)
            switch(opt_algo) {
                case ALGO_SCRYPT:      nv_scrypt_core_kernelA<ALGO_SCRYPT>     <<< grid, threads, 0, stream >>>(d_idata, pos, min(pos+batch, N)); break;
                case ALGO_SCRYPT_JANE: nv_scrypt_core_kernelA<ALGO_SCRYPT_JANE><<< grid, threads, 0, stream >>>(d_idata, pos, min(pos+batch, N)); break;
            }
        else
            switch(opt_algo) {
                case ALGO_SCRYPT:      nv_scrypt_core_kernelA_LG<ALGO_SCRYPT>     <<< grid, threads, 0, stream >>>(d_idata, pos, min(pos+batch, N), LOOKUP_GAP); break;
                case ALGO_SCRYPT_JANE: nv_scrypt_core_kernelA_LG<ALGO_SCRYPT_JANE><<< grid, threads, 0, stream >>>(d_idata, pos, min(pos+batch, N), LOOKUP_GAP); break;
            }

        pos += batch;
    } while (pos < N);

    // Second phase: Random read access from scratchpad.
    pos = 0;
    do
    {
        if (LOOKUP_GAP == 1) {
            if (texture_cache == 0) switch(opt_algo) {
                case ALGO_SCRYPT:      nv_scrypt_core_kernelB<ALGO_SCRYPT     ,0><<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N)); break;
                case ALGO_SCRYPT_JANE: nv_scrypt_core_kernelB<ALGO_SCRYPT_JANE,0><<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N)); break;
            }
            else if (texture_cache == 1) switch(opt_algo) {
                case ALGO_SCRYPT:      nv_scrypt_core_kernelB<ALGO_SCRYPT     ,1><<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N)); break;
                case ALGO_SCRYPT_JANE: nv_scrypt_core_kernelB<ALGO_SCRYPT_JANE,1><<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N)); break;
            }
            else if (texture_cache == 2) switch(opt_algo) {
                case ALGO_SCRYPT:      nv_scrypt_core_kernelB<ALGO_SCRYPT     ,2><<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N)); break;
                case ALGO_SCRYPT_JANE: nv_scrypt_core_kernelB<ALGO_SCRYPT_JANE,2><<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N)); break;
            }
        } else {
            if (texture_cache == 0) switch(opt_algo) {
                case ALGO_SCRYPT:      nv_scrypt_core_kernelB_LG<ALGO_SCRYPT     ,0><<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N), LOOKUP_GAP); break;
                case ALGO_SCRYPT_JANE: nv_scrypt_core_kernelB_LG<ALGO_SCRYPT_JANE,0><<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N), LOOKUP_GAP); break;
            }
            else if (texture_cache == 1) switch(opt_algo) {
                case ALGO_SCRYPT:      nv_scrypt_core_kernelB_LG<ALGO_SCRYPT     ,1><<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N), LOOKUP_GAP); break;
                case ALGO_SCRYPT_JANE: nv_scrypt_core_kernelB_LG<ALGO_SCRYPT_JANE,1><<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N), LOOKUP_GAP); break;
            }
            else if (texture_cache == 2) switch(opt_algo) {
                case ALGO_SCRYPT:      nv_scrypt_core_kernelB_LG<ALGO_SCRYPT     ,2><<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N), LOOKUP_GAP); break;
                case ALGO_SCRYPT_JANE: nv_scrypt_core_kernelB_LG<ALGO_SCRYPT_JANE,2><<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N), LOOKUP_GAP); break;
            }
        }

        pos += batch;
    } while (pos < N);

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

__device__ __forceinline__ uint4 __shfl(const uint4 val, unsigned int lane, unsigned int width)
{
    return make_uint4(
        (unsigned int)__shfl((int)val.x, lane, width),
        (unsigned int)__shfl((int)val.y, lane, width),
        (unsigned int)__shfl((int)val.z, lane, width),
        (unsigned int)__shfl((int)val.w, lane, width));
}

__device__ __forceinline__ void __transposed_write_BC(uint4 (&B)[4], uint4 (&C)[4], uint4 *D, int spacing)
{
    unsigned int laneId = __laneId();

    unsigned int lane8 = laneId%8;
    unsigned int tile  = laneId/8;
    
    uint4 T1[8], T2[8];

    /* Source matrix, A-H are threads, 0-7 are data items, thread A is marked with `*`:

       *A0  B0  C0  D0  E0  F0  G0  H0
       *A1  B1  C1  D1  E1  F1  G1  H1
       *A2  B2  C2  D2  E2  F2  G2  H2
       *A3  B3  C3  D3  E3  F3  G3  H3
       *A4  B4  C4  D4  E4  F4  G4  H4
       *A5  B5  C5  D5  E5  F5  G5  H5
       *A6  B6  C6  D6  E6  F6  G6  H6
       *A7  B7  C7  D7  E7  F7  G7  H7
    */

    // rotate rows
    T1[0] = B[0];
    T1[1] = __shfl(B[1], lane8 + 7, 8);
    T1[2] = __shfl(B[2], lane8 + 6, 8);
    T1[3] = __shfl(B[3], lane8 + 5, 8);
    T1[4] = __shfl(C[0], lane8 + 4, 8);
    T1[5] = __shfl(C[1], lane8 + 3, 8);
    T1[6] = __shfl(C[2], lane8 + 2, 8);
    T1[7] = __shfl(C[3], lane8 + 1, 8);

    /* Matrix after row rotates:

       *A0  B0  C0  D0  E0  F0  G0  H0
        H1 *A1  B1  C1  D1  E1  F1  G1
        G2  H2 *A2  B2  C2  D2  E2  F2
        F3  G3  H3 *A3  B3  C3  D3  E3
        E4  F4  G4  H4 *A4  B4  C4  D4
        D5  E5  F5  G5  H5 *A5  B5  C5
        C6  D6  E6  F6  G6  H6 *A6  B6
        B7  C7  D7  E7  F7  G7  H7 *A7
    */

    // rotate columns up using a barrel shifter simulation
    // column X is rotated up by (X+1) items
#pragma unroll 8
    for(int n = 0; n < 8; n++) T2[n] = ((lane8+1) & 1) ? T1[(n+1) % 8] : T1[n];
#pragma unroll 8
    for(int n = 0; n < 8; n++) T1[n] = ((lane8+1) & 2) ? T2[(n+2) % 8] : T2[n];
#pragma unroll 8
    for(int n = 0; n < 8; n++) T2[n] = ((lane8+1) & 4) ? T1[(n+4) % 8] : T1[n];

    /* Matrix after column rotates:

        H1  H2  H3  H4  H5  H6  H7  H0
        G2  G3  G4  G5  G6  G7  G0  G1   
        F3  F4  F5  F6  F7  F0  F1  F2       
        E4  E5  E6  E7  E0  E1  E2  E3           
        D5  D6  D7  D0  D1  D2  D3  D4               
        C6  C7  C0  C1  C2  C3  C4  C5                   
        B7  B0  B1  B2  B3  B4  B5  B6                       
       *A0 *A1 *A2 *A3 *A4 *A5 *A6 *A7
    */

    // rotate rows again using address math and write to D, in reverse row order
    D[spacing*2*(32*tile   )+ lane8     ] = T2[7];
    D[spacing*2*(32*tile+4 )+(lane8+7)%8] = T2[6];
    D[spacing*2*(32*tile+8 )+(lane8+6)%8] = T2[5];
    D[spacing*2*(32*tile+12)+(lane8+5)%8] = T2[4];
    D[spacing*2*(32*tile+16)+(lane8+4)%8] = T2[3];
    D[spacing*2*(32*tile+20)+(lane8+3)%8] = T2[2];
    D[spacing*2*(32*tile+24)+(lane8+2)%8] = T2[1];
    D[spacing*2*(32*tile+28)+(lane8+1)%8] = T2[0];
}

template <int TEX_DIM> __device__ __forceinline__ void __transposed_read_BC(const uint4 *S, uint4 (&B)[4], uint4 (&C)[4], int spacing, int row)
{
    unsigned int laneId = __laneId();

    unsigned int lane8 = laneId%8;
    unsigned int tile  = laneId/8;

    // Perform the same transposition as in __transposed_write_BC, but in reverse order.
    // See the illustrations in comments for __transposed_write_BC.

    // read and rotate rows, in reverse row order
    uint4 T1[8], T2[8];
    const uint4 *loc;
    loc = &S[(spacing*2*(32*tile   ) +  lane8      + 8*__shfl(row, 0, 8))];
    T1[7] = TEX_DIM==0 ? __ldg(loc) : TEX_DIM==1 ? tex1Dfetch(texRef1D_4_V, loc-(uint4*)c_V[0]) : tex2D(texRef2D_4_V, 0.5f + ((loc-(uint4*)c_V[0])%TEXWIDTH), 0.5f + ((loc-(uint4*)c_V[0])/TEXWIDTH));
    loc = &S[(spacing*2*(32*tile+4 ) + (lane8+7)%8 + 8*__shfl(row, 1, 8))];
    T1[6] = TEX_DIM==0 ? __ldg(loc) : TEX_DIM==1 ? tex1Dfetch(texRef1D_4_V, loc-(uint4*)c_V[0]) : tex2D(texRef2D_4_V, 0.5f + ((loc-(uint4*)c_V[0])%TEXWIDTH), 0.5f + ((loc-(uint4*)c_V[0])/TEXWIDTH));
    loc = &S[(spacing*2*(32*tile+8 ) + (lane8+6)%8 + 8*__shfl(row, 2, 8))];
    T1[5] = TEX_DIM==0 ? __ldg(loc) : TEX_DIM==1 ? tex1Dfetch(texRef1D_4_V, loc-(uint4*)c_V[0]) : tex2D(texRef2D_4_V, 0.5f + ((loc-(uint4*)c_V[0])%TEXWIDTH), 0.5f + ((loc-(uint4*)c_V[0])/TEXWIDTH));
    loc = &S[(spacing*2*(32*tile+12) + (lane8+5)%8 + 8*__shfl(row, 3, 8))];
    T1[4] = TEX_DIM==0 ? __ldg(loc) : TEX_DIM==1 ? tex1Dfetch(texRef1D_4_V, loc-(uint4*)c_V[0]) : tex2D(texRef2D_4_V, 0.5f + ((loc-(uint4*)c_V[0])%TEXWIDTH), 0.5f + ((loc-(uint4*)c_V[0])/TEXWIDTH));
    loc = &S[(spacing*2*(32*tile+16) + (lane8+4)%8 + 8*__shfl(row, 4, 8))];
    T1[3] = TEX_DIM==0 ? __ldg(loc) : TEX_DIM==1 ? tex1Dfetch(texRef1D_4_V, loc-(uint4*)c_V[0]) : tex2D(texRef2D_4_V, 0.5f + ((loc-(uint4*)c_V[0])%TEXWIDTH), 0.5f + ((loc-(uint4*)c_V[0])/TEXWIDTH));
    loc = &S[(spacing*2*(32*tile+20) + (lane8+3)%8 + 8*__shfl(row, 5, 8))];
    T1[2] = TEX_DIM==0 ? __ldg(loc) : TEX_DIM==1 ? tex1Dfetch(texRef1D_4_V, loc-(uint4*)c_V[0]) : tex2D(texRef2D_4_V, 0.5f + ((loc-(uint4*)c_V[0])%TEXWIDTH), 0.5f + ((loc-(uint4*)c_V[0])/TEXWIDTH));
    loc = &S[(spacing*2*(32*tile+24) + (lane8+2)%8 + 8*__shfl(row, 6, 8))];
    T1[1] = TEX_DIM==0 ? __ldg(loc) : TEX_DIM==1 ? tex1Dfetch(texRef1D_4_V, loc-(uint4*)c_V[0]) : tex2D(texRef2D_4_V, 0.5f + ((loc-(uint4*)c_V[0])%TEXWIDTH), 0.5f + ((loc-(uint4*)c_V[0])/TEXWIDTH));
    loc = &S[(spacing*2*(32*tile+28) + (lane8+1)%8 + 8*__shfl(row, 7, 8))];
    T1[0] = TEX_DIM==0 ? __ldg(loc) : TEX_DIM==1 ? tex1Dfetch(texRef1D_4_V, loc-(uint4*)c_V[0]) : tex2D(texRef2D_4_V, 0.5f + ((loc-(uint4*)c_V[0])%TEXWIDTH), 0.5f + ((loc-(uint4*)c_V[0])/TEXWIDTH));
    
    // rotate columns down using a barrel shifter simulation
    // column X is rotated down by (X+1) items, or up by (8-(X+1)) = (7-X) items
#pragma unroll 8
    for(int n = 0; n < 8; n++) T2[n] = ((7-lane8) & 1) ? T1[(n+1) % 8] : T1[n];
#pragma unroll 8
    for(int n = 0; n < 8; n++) T1[n] = ((7-lane8) & 2) ? T2[(n+2) % 8] : T2[n];
#pragma unroll 8
    for(int n = 0; n < 8; n++) T2[n] = ((7-lane8) & 4) ? T1[(n+4) % 8] : T1[n];
    
    // rotate rows
    B[0] = T2[0];
    B[1] = __shfl(T2[1], lane8 + 1, 8);
    B[2] = __shfl(T2[2], lane8 + 2, 8);
    B[3] = __shfl(T2[3], lane8 + 3, 8);
    C[0] = __shfl(T2[4], lane8 + 4, 8);
    C[1] = __shfl(T2[5], lane8 + 5, 8);
    C[2] = __shfl(T2[6], lane8 + 6, 8);
    C[3] = __shfl(T2[7], lane8 + 7, 8);

}

template <int TEX_DIM> __device__ __forceinline__ void __transposed_xor_BC(const uint4 *S, uint4 (&B)[4], uint4 (&C)[4], int spacing, int row)
{
    uint4 BT[4], CT[4];
    __transposed_read_BC<TEX_DIM>(S, BT, CT, spacing, row);

#pragma unroll 4
    for(int n = 0; n < 4; n++) 
    {
        B[n] ^= BT[n];
        C[n] ^= CT[n];
    }
}

#if __CUDA_ARCH__ < 350 
    // Kepler (Compute 3.0)
    #define ROTL(a, b) ((a)<<(b))|((a)>>(32-(b)))
#else
    // Kepler (Compute 3.5)
    #define ROTL(a, b) __funnelshift_l( a, a, b );
#endif



#if 0

#define QUARTER(a,b,c,d) \
    a += b; d ^= a; d = ROTL(d,16); \
    c += d; b ^= c; b = ROTL(b,12); \
    a += b; d ^= a; d = ROTL(d,8); \
    c += d; b ^= c; b = ROTL(b,7);

static __device__ void xor_chacha8(uint4 *B, uint4 *C)
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
    QUARTER( x[0], x[4], x[ 8], x[12] )
    QUARTER( x[1], x[5], x[ 9], x[13] )
    QUARTER( x[2], x[6], x[10], x[14] )
    QUARTER( x[3], x[7], x[11], x[15] )

    /* Operate on diagonals */
    QUARTER( x[0], x[5], x[10], x[15] )
    QUARTER( x[1], x[6], x[11], x[12] )
    QUARTER( x[2], x[7], x[ 8], x[13] )
    QUARTER( x[3], x[4], x[ 9], x[14] )

    /* Operate on columns. */
    QUARTER( x[0], x[4], x[ 8], x[12] )
    QUARTER( x[1], x[5], x[ 9], x[13] )
    QUARTER( x[2], x[6], x[10], x[14] )
    QUARTER( x[3], x[7], x[11], x[15] )

    /* Operate on diagonals */
    QUARTER( x[0], x[5], x[10], x[15] )
    QUARTER( x[1], x[6], x[11], x[12] )
    QUARTER( x[2], x[7], x[ 8], x[13] )
    QUARTER( x[3], x[4], x[ 9], x[14] )

    /* Operate on columns. */
    QUARTER( x[0], x[4], x[ 8], x[12] )
    QUARTER( x[1], x[5], x[ 9], x[13] )
    QUARTER( x[2], x[6], x[10], x[14] )
    QUARTER( x[3], x[7], x[11], x[15] )

    /* Operate on diagonals */
    QUARTER( x[0], x[5], x[10], x[15] )
    QUARTER( x[1], x[6], x[11], x[12] )
    QUARTER( x[2], x[7], x[ 8], x[13] )
    QUARTER( x[3], x[4], x[ 9], x[14] )

    /* Operate on columns. */
    QUARTER( x[0], x[4], x[ 8], x[12] )
    QUARTER( x[1], x[5], x[ 9], x[13] )
    QUARTER( x[2], x[6], x[10], x[14] )
    QUARTER( x[3], x[7], x[11], x[15] )

    /* Operate on diagonals */
    QUARTER( x[0], x[5], x[10], x[15] )
    QUARTER( x[1], x[6], x[11], x[12] )
    QUARTER( x[2], x[7], x[ 8], x[13] )
    QUARTER( x[3], x[4], x[ 9], x[14] )

    B[0].x += x[0]; B[0].y += x[1]; B[0].z += x[2];  B[0].w += x[3];  B[1].x += x[4];  B[1].y += x[5];  B[1].z += x[6];  B[1].w += x[7];
    B[2].x += x[8]; B[2].y += x[9]; B[2].z += x[10]; B[2].w += x[11]; B[3].x += x[12]; B[3].y += x[13]; B[3].z += x[14]; B[3].w += x[15];
}

#else

#define ADD4(d1,d2,d3,d4,s1,s2,s3,s4) \
    d1 += s1; d2 += s2; d3 += s3; d4 += s4;

#define XOR4(d1,d2,d3,d4,s1,s2,s3,s4) \
    d1 ^= s1; d2 ^= s2; d3 ^= s3; d4 ^= s4;

#define ROTL4(d1,d2,d3,d4,amt) \
    d1 = ROTL(d1, amt); d2 = ROTL(d2, amt); d3 = ROTL(d3, amt); d4 = ROTL(d4, amt);

#define QROUND(a1,a2,a3,a4, b1,b2,b3,b4, c1,c2,c3,c4, amt) \
    ADD4 (a1,a2,a3,a4, c1,c2,c3,c4) \
    XOR4 (b1,b2,b3,b4, a1,a2,a3,a4) \
    ROTL4(b1,b2,b3,b4, amt)

static __device__ void xor_chacha8(uint4 *B, uint4 *C)
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
    QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[12],x[13],x[14],x[15], x[ 4],x[ 5],x[ 6],x[ 7], 16);
    QROUND(x[ 8],x[ 9],x[10],x[11], x[ 4],x[ 5],x[ 6],x[ 7], x[12],x[13],x[14],x[15], 12);
    QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[12],x[13],x[14],x[15], x[ 4],x[ 5],x[ 6],x[ 7],  8);
    QROUND(x[ 8],x[ 9],x[10],x[11], x[ 4],x[ 5],x[ 6],x[ 7], x[12],x[13],x[14],x[15],  7);

    /* Operate on diagonals */
    QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[15],x[12],x[13],x[14], x[ 5],x[ 6],x[ 7],x[ 4], 16);
    QROUND(x[10],x[11],x[ 8],x[ 9], x[ 5],x[ 6],x[ 7],x[ 4], x[15],x[12],x[13],x[14], 12);
    QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[15],x[12],x[13],x[14], x[ 5],x[ 6],x[ 7],x[ 4],  8);
    QROUND(x[10],x[11],x[ 8],x[ 9], x[ 5],x[ 6],x[ 7],x[ 4], x[15],x[12],x[13],x[14],  7);

    /* Operate on columns. */
    QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[12],x[13],x[14],x[15], x[ 4],x[ 5],x[ 6],x[ 7], 16);
    QROUND(x[ 8],x[ 9],x[10],x[11], x[ 4],x[ 5],x[ 6],x[ 7], x[12],x[13],x[14],x[15], 12);
    QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[12],x[13],x[14],x[15], x[ 4],x[ 5],x[ 6],x[ 7],  8);
    QROUND(x[ 8],x[ 9],x[10],x[11], x[ 4],x[ 5],x[ 6],x[ 7], x[12],x[13],x[14],x[15],  7);

    /* Operate on diagonals */
    QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[15],x[12],x[13],x[14], x[ 5],x[ 6],x[ 7],x[ 4], 16);
    QROUND(x[10],x[11],x[ 8],x[ 9], x[ 5],x[ 6],x[ 7],x[ 4], x[15],x[12],x[13],x[14], 12);
    QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[15],x[12],x[13],x[14], x[ 5],x[ 6],x[ 7],x[ 4],  8);
    QROUND(x[10],x[11],x[ 8],x[ 9], x[ 5],x[ 6],x[ 7],x[ 4], x[15],x[12],x[13],x[14],  7);

    /* Operate on columns. */
    QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[12],x[13],x[14],x[15], x[ 4],x[ 5],x[ 6],x[ 7], 16);
    QROUND(x[ 8],x[ 9],x[10],x[11], x[ 4],x[ 5],x[ 6],x[ 7], x[12],x[13],x[14],x[15], 12);
    QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[12],x[13],x[14],x[15], x[ 4],x[ 5],x[ 6],x[ 7],  8);
    QROUND(x[ 8],x[ 9],x[10],x[11], x[ 4],x[ 5],x[ 6],x[ 7], x[12],x[13],x[14],x[15],  7);

    /* Operate on diagonals */
    QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[15],x[12],x[13],x[14], x[ 5],x[ 6],x[ 7],x[ 4], 16);
    QROUND(x[10],x[11],x[ 8],x[ 9], x[ 5],x[ 6],x[ 7],x[ 4], x[15],x[12],x[13],x[14], 12);
    QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[15],x[12],x[13],x[14], x[ 5],x[ 6],x[ 7],x[ 4],  8);
    QROUND(x[10],x[11],x[ 8],x[ 9], x[ 5],x[ 6],x[ 7],x[ 4], x[15],x[12],x[13],x[14],  7);

    /* Operate on columns. */
    QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[12],x[13],x[14],x[15], x[ 4],x[ 5],x[ 6],x[ 7], 16);
    QROUND(x[ 8],x[ 9],x[10],x[11], x[ 4],x[ 5],x[ 6],x[ 7], x[12],x[13],x[14],x[15], 12);
    QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[12],x[13],x[14],x[15], x[ 4],x[ 5],x[ 6],x[ 7],  8);
    QROUND(x[ 8],x[ 9],x[10],x[11], x[ 4],x[ 5],x[ 6],x[ 7], x[12],x[13],x[14],x[15],  7);

    /* Operate on diagonals */
    QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[15],x[12],x[13],x[14], x[ 5],x[ 6],x[ 7],x[ 4], 16);
    QROUND(x[10],x[11],x[ 8],x[ 9], x[ 5],x[ 6],x[ 7],x[ 4], x[15],x[12],x[13],x[14], 12);
    QROUND(x[ 0],x[ 1],x[ 2],x[ 3], x[15],x[12],x[13],x[14], x[ 5],x[ 6],x[ 7],x[ 4],  8);
    QROUND(x[10],x[11],x[ 8],x[ 9], x[ 5],x[ 6],x[ 7],x[ 4], x[15],x[12],x[13],x[14],  7);

    B[0].x += x[0]; B[0].y += x[1]; B[0].z += x[2];  B[0].w += x[3];  B[1].x += x[4];  B[1].y += x[5];  B[1].z += x[6];  B[1].w += x[7];
    B[2].x += x[8]; B[2].y += x[9]; B[2].z += x[10]; B[2].w += x[11]; B[3].x += x[12]; B[3].y += x[13]; B[3].z += x[14]; B[3].w += x[15];
}

#endif


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


template <int ALGO> static __device__ void block_mixer(uint4 *B, uint4 *C)
{
  switch (ALGO)
  {
    case ALGO_SCRYPT:      xor_salsa8(B, C); break;
    case ALGO_SCRYPT_JANE: xor_chacha8(B, C); break;
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Experimental Scrypt core kernel for Kepler devices.
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
template <int ALGO> __global__ void nv_scrypt_core_kernelA(uint32_t *g_idata, int begin, int end)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x / warpSize * warpSize;
    g_idata += 32 * offset;
    uint32_t * V = c_V[offset / warpSize];
    uint4 B[4], C[4];
    int i = begin;

    if(i == 0) {
        __transposed_read_BC<0>((uint4*)g_idata, B, C, 1, 0);
        __transposed_write_BC(B, C, (uint4*)V, c_N); 
        ++i;
    } else
        __transposed_read_BC<0>((uint4*)(V + (i-1)*32), B, C, c_N, 0);

    while(i < end) {
        block_mixer<ALGO>(B, C); block_mixer<ALGO>(C, B);
        __transposed_write_BC(B, C, (uint4*)(V + i*32), c_N); 
        ++i;
    }
}

template <int ALGO> __global__ void nv_scrypt_core_kernelA_LG(uint32_t *g_idata, int begin, int end, unsigned int LOOKUP_GAP)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x / warpSize * warpSize;
    g_idata += 32 * offset;
    uint32_t * V = c_V[offset / warpSize];
    uint4 B[4], C[4];
    int i = begin;

    if(i == 0) {
        __transposed_read_BC<0>((uint4*)g_idata, B, C, 1, 0);
        __transposed_write_BC(B, C, (uint4*)V, c_spacing); 
        ++i;
    } else {
        int pos = (i-1)/LOOKUP_GAP, loop = (i-1)-pos*LOOKUP_GAP;
        __transposed_read_BC<0>((uint4*)(V + pos*32), B, C, c_spacing, 0);
        while(loop--) { block_mixer<ALGO>(B, C); block_mixer<ALGO>(C, B); }
    }

    while(i < end) {
        block_mixer<ALGO>(B, C); block_mixer<ALGO>(C, B);
        if (i % LOOKUP_GAP == 0)
          __transposed_write_BC(B, C, (uint4*)(V + (i/LOOKUP_GAP)*32), c_spacing); 
        ++i;
    }
}

template <int ALGO, int TEX_DIM> __global__ void nv_scrypt_core_kernelB(uint32_t *g_odata, int begin, int end)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x / warpSize * warpSize;
    g_odata += 32 * offset;
    uint32_t * V = c_V[offset / warpSize];
    uint4 B[4], C[4];

    if(begin == 0) {
        __transposed_read_BC<TEX_DIM>((uint4*)V, B, C, c_N, c_N_1);
        block_mixer<ALGO>(B, C); block_mixer<ALGO>(C, B);
    } else
        __transposed_read_BC<0>((uint4*)g_odata, B, C, 1, 0);

    for (int i = begin; i < end; i++)  {
        int slot = C[0].x & c_N_1;
        __transposed_xor_BC<TEX_DIM>((uint4*)(V), B, C, c_N, slot);
        block_mixer<ALGO>(B, C); block_mixer<ALGO>(C, B);
    }

    __transposed_write_BC(B, C, (uint4*)(g_odata), 1);
}

template <int ALGO, int TEX_DIM> __global__ void nv_scrypt_core_kernelB_LG(uint32_t *g_odata, int begin, int end, unsigned int LOOKUP_GAP)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x / warpSize * warpSize;
    g_odata += 32 * offset;
    uint32_t * V = c_V[offset / warpSize];
    uint4 B[4], C[4];

    if(begin == 0) {
      int pos = c_N_1/LOOKUP_GAP, loop = 1 + (c_N_1-pos*LOOKUP_GAP);
      __transposed_read_BC<TEX_DIM>((uint4*)V, B, C, c_spacing, pos);
      while(loop--) { block_mixer<ALGO>(B, C); block_mixer<ALGO>(C, B); }
    } else {
        __transposed_read_BC<TEX_DIM>((uint4*)g_odata, B, C, 1, 0);
    }

    for (int i = begin; i < end; i++)  {
        int slot = C[0].x & c_N_1;
        int pos = slot/LOOKUP_GAP, loop = slot-pos*LOOKUP_GAP;
        uint4 b[4], c[4]; __transposed_read_BC<TEX_DIM>((uint4*)(V), b, c, c_spacing, pos);
        while(loop--) { block_mixer<ALGO>(b, c); block_mixer<ALGO>(c, b); }
#pragma unroll 4
        for(int n = 0; n < 4; n++) { B[n] ^= b[n]; C[n] ^= c[n]; }
        block_mixer<ALGO>(B, C); block_mixer<ALGO>(C, B);
    }

    __transposed_write_BC(B, C, (uint4*)(g_odata), 1);
}



//
// Maxcoin related Keccak implementation (Keccak256)
//

// from salsa_kernel.cu
extern std::map<int, int> context_blocks;
extern std::map<int, int> context_wpb;
extern std::map<int, KernelInterface *> context_kernel;
extern std::map<int, cudaStream_t> context_streams[2];
extern std::map<int, uint32_t *> context_hash[2];

__constant__ uint64_t ptarget64[4];

#define ROL(a, offset) ((((uint64_t)a) << ((offset) % 64)) ^ (((uint64_t)a) >> (64-((offset) % 64))))
#define ROL_mult8(a, offset) ROL(a, offset)

__constant__ uint64_t KeccakF_RoundConstants[24];

static uint64_t host_KeccakF_RoundConstants[24] = 
{
    (uint64_t)0x0000000000000001ULL,
    (uint64_t)0x0000000000008082ULL,
    (uint64_t)0x800000000000808aULL,
    (uint64_t)0x8000000080008000ULL,
    (uint64_t)0x000000000000808bULL,
    (uint64_t)0x0000000080000001ULL,
    (uint64_t)0x8000000080008081ULL,
    (uint64_t)0x8000000000008009ULL,
    (uint64_t)0x000000000000008aULL,
    (uint64_t)0x0000000000000088ULL,
    (uint64_t)0x0000000080008009ULL,
    (uint64_t)0x000000008000000aULL,
    (uint64_t)0x000000008000808bULL,
    (uint64_t)0x800000000000008bULL,
    (uint64_t)0x8000000000008089ULL,
    (uint64_t)0x8000000000008003ULL,
    (uint64_t)0x8000000000008002ULL,
    (uint64_t)0x8000000000000080ULL,
    (uint64_t)0x000000000000800aULL,
    (uint64_t)0x800000008000000aULL,
    (uint64_t)0x8000000080008081ULL,
    (uint64_t)0x8000000000008080ULL,
    (uint64_t)0x0000000080000001ULL,
    (uint64_t)0x8000000080008008ULL
};

__constant__ uint64_t pdata64[10];

static __device__ uint32_t cuda_swab32(uint32_t x)
{
    return (((x << 24) & 0xff000000u) | ((x << 8) & 0x00ff0000u)
          | ((x >> 8) & 0x0000ff00u) | ((x >> 24) & 0x000000ffu));
}

__global__ void kepler_crypto_hash( uint64_t *g_out, uint32_t nonce, uint32_t *g_good, bool validate )
{
    uint64_t Aba, Abe, Abi, Abo, Abu;
    uint64_t Aga, Age, Agi, Ago, Agu;
    uint64_t Aka, Ake, Aki, Ako, Aku;
    uint64_t Ama, Ame, Ami, Amo, Amu;
    uint64_t Asa, Ase, Asi, Aso, Asu;
    uint64_t BCa, BCe, BCi, BCo, BCu;
    uint64_t Da, De, Di, Do, Du;
    uint64_t Eba, Ebe, Ebi, Ebo, Ebu;
    uint64_t Ega, Ege, Egi, Ego, Egu;
    uint64_t Eka, Eke, Eki, Eko, Eku;
    uint64_t Ema, Eme, Emi, Emo, Emu;
    uint64_t Esa, Ese, Esi, Eso, Esu;

    //copyFromState(A, state)
    Aba = pdata64[0];
    Abe = pdata64[1];
    Abi = pdata64[2];
    Abo = pdata64[3];
    Abu = pdata64[4];
    Aga = pdata64[5];
    Age = pdata64[6];
    Agi = pdata64[7];
    Ago = pdata64[8];
    Agu = (pdata64[9] & 0x00000000FFFFFFFFULL) | (((uint64_t)cuda_swab32(nonce + ((blockIdx.x * blockDim.x) + threadIdx.x))) << 32);
    Aka = 0x0000000000000001ULL;
    Ake = 0;
    Aki = 0;
    Ako = 0;
    Aku = 0;
    Ama = 0;
    Ame = 0x8000000000000000ULL;
    Ami = 0;
    Amo = 0;
    Amu = 0;
    Asa = 0;
    Ase = 0;
    Asi = 0;
    Aso = 0;
    Asu = 0;

#pragma unroll 12
    for( int laneCount = 0; laneCount < 24; laneCount += 2 )
    {
        //    prepareTheta
        BCa = Aba^Aga^Aka^Ama^Asa;
        BCe = Abe^Age^Ake^Ame^Ase;
        BCi = Abi^Agi^Aki^Ami^Asi;
        BCo = Abo^Ago^Ako^Amo^Aso;
        BCu = Abu^Agu^Aku^Amu^Asu;

        //thetaRhoPiChiIotaPrepareTheta(round  , A, E)
        Da = BCu^ROL(BCe, 1);
        De = BCa^ROL(BCi, 1);
        Di = BCe^ROL(BCo, 1);
        Do = BCi^ROL(BCu, 1);
        Du = BCo^ROL(BCa, 1);

        Aba ^= Da;
        BCa = Aba;
        Age ^= De;
        BCe = ROL(Age, 44);
        Aki ^= Di;
        BCi = ROL(Aki, 43);
        Amo ^= Do;
        BCo = ROL(Amo, 21);
        Asu ^= Du;
        BCu = ROL(Asu, 14);
        Eba =   BCa ^((~BCe)&  BCi );
        Eba ^= (uint64_t)KeccakF_RoundConstants[laneCount];
        Ebe =   BCe ^((~BCi)&  BCo );
        Ebi =   BCi ^((~BCo)&  BCu );
        Ebo =   BCo ^((~BCu)&  BCa );
        Ebu =   BCu ^((~BCa)&  BCe );

        Abo ^= Do;
        BCa = ROL(Abo, 28);
        Agu ^= Du;
        BCe = ROL(Agu, 20);
        Aka ^= Da;
        BCi = ROL(Aka,  3);
        Ame ^= De;
        BCo = ROL(Ame, 45);
        Asi ^= Di;
        BCu = ROL(Asi, 61);
        Ega =   BCa ^((~BCe)&  BCi );
        Ege =   BCe ^((~BCi)&  BCo );
        Egi =   BCi ^((~BCo)&  BCu );
        Ego =   BCo ^((~BCu)&  BCa );
        Egu =   BCu ^((~BCa)&  BCe );

        Abe ^= De;
        BCa = ROL(Abe,  1);
        Agi ^= Di;
        BCe = ROL(Agi,  6);
        Ako ^= Do;
        BCi = ROL(Ako, 25);
        Amu ^= Du;
        BCo = ROL_mult8(Amu,  8);
        Asa ^= Da;
        BCu = ROL(Asa, 18);
        Eka =   BCa ^((~BCe)&  BCi );
        Eke =   BCe ^((~BCi)&  BCo );
        Eki =   BCi ^((~BCo)&  BCu );
        Eko =   BCo ^((~BCu)&  BCa );
        Eku =   BCu ^((~BCa)&  BCe );

        Abu ^= Du;
        BCa = ROL(Abu, 27);
        Aga ^= Da;
        BCe = ROL(Aga, 36);
        Ake ^= De;
        BCi = ROL(Ake, 10);
        Ami ^= Di;
        BCo = ROL(Ami, 15);
        Aso ^= Do;
        BCu = ROL_mult8(Aso, 56);
        Ema =   BCa ^((~BCe)&  BCi );
        Eme =   BCe ^((~BCi)&  BCo );
        Emi =   BCi ^((~BCo)&  BCu );
        Emo =   BCo ^((~BCu)&  BCa );
        Emu =   BCu ^((~BCa)&  BCe );

        Abi ^= Di;
        BCa = ROL(Abi, 62);
        Ago ^= Do;
        BCe = ROL(Ago, 55);
        Aku ^= Du;
        BCi = ROL(Aku, 39);
        Ama ^= Da;
        BCo = ROL(Ama, 41);
        Ase ^= De;
        BCu = ROL(Ase,  2);
        Esa =   BCa ^((~BCe)&  BCi );
        Ese =   BCe ^((~BCi)&  BCo );
        Esi =   BCi ^((~BCo)&  BCu );
        Eso =   BCo ^((~BCu)&  BCa );
        Esu =   BCu ^((~BCa)&  BCe );

        //    prepareTheta
        BCa = Eba^Ega^Eka^Ema^Esa;
        BCe = Ebe^Ege^Eke^Eme^Ese;
        BCi = Ebi^Egi^Eki^Emi^Esi;
        BCo = Ebo^Ego^Eko^Emo^Eso;
        BCu = Ebu^Egu^Eku^Emu^Esu;

        //thetaRhoPiChiIotaPrepareTheta(round+1, E, A)
        Da = BCu^ROL(BCe, 1);
        De = BCa^ROL(BCi, 1);
        Di = BCe^ROL(BCo, 1);
        Do = BCi^ROL(BCu, 1);
        Du = BCo^ROL(BCa, 1);

        Eba ^= Da;
        BCa = Eba;
        Ege ^= De;
        BCe = ROL(Ege, 44);
        Eki ^= Di;
        BCi = ROL(Eki, 43);
        Emo ^= Do;
        BCo = ROL(Emo, 21);
        Esu ^= Du;
        BCu = ROL(Esu, 14);
        Aba =   BCa ^((~BCe)&  BCi );
        Aba ^= (uint64_t)KeccakF_RoundConstants[laneCount+1];
        Abe =   BCe ^((~BCi)&  BCo );
        Abi =   BCi ^((~BCo)&  BCu );
        Abo =   BCo ^((~BCu)&  BCa );
        Abu =   BCu ^((~BCa)&  BCe );

        Ebo ^= Do;
        BCa = ROL(Ebo, 28);
        Egu ^= Du;
        BCe = ROL(Egu, 20);
        Eka ^= Da;
        BCi = ROL(Eka, 3);
        Eme ^= De;
        BCo = ROL(Eme, 45);
        Esi ^= Di;
        BCu = ROL(Esi, 61);
        Aga =   BCa ^((~BCe)&  BCi );
        Age =   BCe ^((~BCi)&  BCo );
        Agi =   BCi ^((~BCo)&  BCu );
        Ago =   BCo ^((~BCu)&  BCa );
        Agu =   BCu ^((~BCa)&  BCe );

        Ebe ^= De;
        BCa = ROL(Ebe, 1);
        Egi ^= Di;
        BCe = ROL(Egi, 6);
        Eko ^= Do;
        BCi = ROL(Eko, 25);
        Emu ^= Du;
        BCo = ROL_mult8(Emu, 8);
        Esa ^= Da;
        BCu = ROL(Esa, 18);
        Aka =   BCa ^((~BCe)&  BCi );
        Ake =   BCe ^((~BCi)&  BCo );
        Aki =   BCi ^((~BCo)&  BCu );
        Ako =   BCo ^((~BCu)&  BCa );
        Aku =   BCu ^((~BCa)&  BCe );

        Ebu ^= Du;
        BCa = ROL(Ebu, 27);
        Ega ^= Da;
        BCe = ROL(Ega, 36);
        Eke ^= De;
        BCi = ROL(Eke, 10);
        Emi ^= Di;
        BCo = ROL(Emi, 15);
        Eso ^= Do;
        BCu = ROL_mult8(Eso, 56);
        Ama =   BCa ^((~BCe)&  BCi );
        Ame =   BCe ^((~BCi)&  BCo );
        Ami =   BCi ^((~BCo)&  BCu );
        Amo =   BCo ^((~BCu)&  BCa );
        Amu =   BCu ^((~BCa)&  BCe );

        Ebi ^= Di;
        BCa = ROL(Ebi, 62);
        Ego ^= Do;
        BCe = ROL(Ego, 55);
        Eku ^= Du;
        BCi = ROL(Eku, 39);
        Ema ^= Da;
        BCo = ROL(Ema, 41);
        Ese ^= De;
        BCu = ROL(Ese, 2);
        Asa =   BCa ^((~BCe)&  BCi );
        Ase =   BCe ^((~BCi)&  BCo );
        Asi =   BCi ^((~BCo)&  BCu );
        Aso =   BCo ^((~BCu)&  BCa );
        Asu =   BCu ^((~BCa)&  BCe );
    }

    if (validate) {
        g_out += 4 * ((blockIdx.x * blockDim.x) + threadIdx.x);
        g_out[3] = Abo;
        g_out[2] = Abi;
        g_out[1] = Abe;
        g_out[0] = Aba;
    }
    
    // the likelyhood of meeting the hashing target is so low, that we're not guarding this
    // with atomic writes, locks or similar...
    uint64_t *g_good64 = (uint64_t*)g_good;
    if (Abo <=  ptarget64[3]) {
        if (Abo < g_good64[3]) {
            g_good64[3] = Abo;
            g_good64[2] = Abi;
            g_good64[1] = Abe;
            g_good64[0] = Aba;
            g_good[8] = nonce + ((blockIdx.x * blockDim.x) + threadIdx.x);
        }
    }
}

static std::map<int, uint32_t *> context_good[2];

bool NVKernel::prepare_keccak256(int thr_id, const uint32_t host_pdata[20], const uint32_t host_ptarget[8])
{
    static bool init[MAX_DEVICES] = {false};
    if (!init[thr_id])
    {
        checkCudaErrors(cudaMemcpyToSymbol(KeccakF_RoundConstants, host_KeccakF_RoundConstants, sizeof(host_KeccakF_RoundConstants), 0, cudaMemcpyHostToDevice));

        // allocate pinned host memory for good hashes
        uint32_t *tmp;
        checkCudaErrors(cudaMalloc((void **) &tmp, 9*sizeof(uint32_t))); context_good[0][thr_id] = tmp;
        checkCudaErrors(cudaMalloc((void **) &tmp, 9*sizeof(uint32_t))); context_good[1][thr_id] = tmp;

        init[thr_id] = true;
    }
    checkCudaErrors(cudaMemcpyToSymbol(pdata64, host_pdata, 20*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(ptarget64, host_ptarget, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

    return context_good[0][thr_id] && context_good[1][thr_id];
}

void NVKernel::do_keccak256(dim3 grid, dim3 threads, int thr_id, int stream, uint32_t *hash, uint32_t nonce, int throughput, bool do_d2h)
{
    checkCudaErrors(cudaMemsetAsync(context_good[stream][thr_id], 0xff, 9 * sizeof(uint32_t), context_streams[stream][thr_id]));

    kepler_crypto_hash<<<grid, threads, 0, context_streams[stream][thr_id]>>>((uint64_t*)context_hash[stream][thr_id], nonce, context_good[stream][thr_id], do_d2h);

    // copy hashes from device memory to host (ALL hashes, lots of data...)
    if (do_d2h && hash != NULL) {
        size_t mem_size = throughput * sizeof(uint32_t) * 8;
        checkCudaErrors(cudaMemcpyAsync(hash, context_hash[stream][thr_id], mem_size,
                        cudaMemcpyDeviceToHost, context_streams[stream][thr_id]));
    }
    else if (hash != NULL) {
        // asynchronous copy of winning nonce (just 4 bytes...)
        checkCudaErrors(cudaMemcpyAsync(hash, context_good[stream][thr_id]+8, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, context_streams[stream][thr_id]));
    }
}


//
// Blakecoin related Keccak implementation (Keccak256)
//

typedef uint32_t sph_u32;
#define SPH_C32(x) ((sph_u32)(x))
#define SPH_T32(x) ((x) & SPH_C32(0xFFFFFFFF))
#if __CUDA_ARCH__ < 350 
    // Kepler (Compute 3.0)
    #define SPH_ROTL32(a, b) ((a)<<(b))|((a)>>(32-(b)))
#else
    // Kepler (Compute 3.5)
    #define SPH_ROTL32(a, b) __funnelshift_l( a, a, b );
#endif
#define SPH_ROTR32(x, n)   SPH_ROTL32(x, (32 - (n)))

__constant__ uint32_t pdata[20];

#ifdef _MSC_VER
#pragma warning (disable: 4146)
#endif

static __device__ sph_u32 cuda_sph_bswap32(sph_u32 x)
{
    return (((x << 24) & 0xff000000u) | ((x << 8) & 0x00ff0000u)
          | ((x >> 8) & 0x0000ff00u) | ((x >> 24) & 0x000000ffu));
}

/**
 * Encode a 32-bit value into the provided buffer (big endian convention).
 *
 * @param dst   the destination buffer
 * @param val   the 32-bit value to encode
 */
static __device__ void
cuda_sph_enc32be(void *dst, sph_u32 val)
{
    *(sph_u32 *)dst = cuda_sph_bswap32(val);
}

#define Z00   0
#define Z01   1
#define Z02   2
#define Z03   3
#define Z04   4
#define Z05   5
#define Z06   6
#define Z07   7
#define Z08   8
#define Z09   9
#define Z0A   A
#define Z0B   B
#define Z0C   C
#define Z0D   D
#define Z0E   E
#define Z0F   F

#define Z10   E
#define Z11   A
#define Z12   4
#define Z13   8
#define Z14   9
#define Z15   F
#define Z16   D
#define Z17   6
#define Z18   1
#define Z19   C
#define Z1A   0
#define Z1B   2
#define Z1C   B
#define Z1D   7
#define Z1E   5
#define Z1F   3

#define Z20   B
#define Z21   8
#define Z22   C
#define Z23   0
#define Z24   5
#define Z25   2
#define Z26   F
#define Z27   D
#define Z28   A
#define Z29   E
#define Z2A   3
#define Z2B   6
#define Z2C   7
#define Z2D   1
#define Z2E   9
#define Z2F   4

#define Z30   7
#define Z31   9
#define Z32   3
#define Z33   1
#define Z34   D
#define Z35   C
#define Z36   B
#define Z37   E
#define Z38   2
#define Z39   6
#define Z3A   5
#define Z3B   A
#define Z3C   4
#define Z3D   0
#define Z3E   F
#define Z3F   8

#define Z40   9
#define Z41   0
#define Z42   5
#define Z43   7
#define Z44   2
#define Z45   4
#define Z46   A
#define Z47   F
#define Z48   E
#define Z49   1
#define Z4A   B
#define Z4B   C
#define Z4C   6
#define Z4D   8
#define Z4E   3
#define Z4F   D

#define Z50   2
#define Z51   C
#define Z52   6
#define Z53   A
#define Z54   0
#define Z55   B
#define Z56   8
#define Z57   3
#define Z58   4
#define Z59   D
#define Z5A   7
#define Z5B   5
#define Z5C   F
#define Z5D   E
#define Z5E   1
#define Z5F   9

#define Z60   C
#define Z61   5
#define Z62   1
#define Z63   F
#define Z64   E
#define Z65   D
#define Z66   4
#define Z67   A
#define Z68   0
#define Z69   7
#define Z6A   6
#define Z6B   3
#define Z6C   9
#define Z6D   2
#define Z6E   8
#define Z6F   B

#define Z70   D
#define Z71   B
#define Z72   7
#define Z73   E
#define Z74   C
#define Z75   1
#define Z76   3
#define Z77   9
#define Z78   5
#define Z79   0
#define Z7A   F
#define Z7B   4
#define Z7C   8
#define Z7D   6
#define Z7E   2
#define Z7F   A

#define Z80   6
#define Z81   F
#define Z82   E
#define Z83   9
#define Z84   B
#define Z85   3
#define Z86   0
#define Z87   8
#define Z88   C
#define Z89   2
#define Z8A   D
#define Z8B   7
#define Z8C   1
#define Z8D   4
#define Z8E   A
#define Z8F   5

#define Z90   A
#define Z91   2
#define Z92   8
#define Z93   4
#define Z94   7
#define Z95   6
#define Z96   1
#define Z97   5
#define Z98   F
#define Z99   B
#define Z9A   9
#define Z9B   E
#define Z9C   3
#define Z9D   C
#define Z9E   D
#define Z9F   0

#define Mx(r, i)    Mx_(Z ## r ## i)
#define Mx_(n)      Mx__(n)
#define Mx__(n)     M ## n

#define CSx(r, i)   CSx_(Z ## r ## i)
#define CSx_(n)     CSx__(n)
#define CSx__(n)    CS ## n

#define CS0   SPH_C32(0x243F6A88)
#define CS1   SPH_C32(0x85A308D3)
#define CS2   SPH_C32(0x13198A2E)
#define CS3   SPH_C32(0x03707344)
#define CS4   SPH_C32(0xA4093822)
#define CS5   SPH_C32(0x299F31D0)
#define CS6   SPH_C32(0x082EFA98)
#define CS7   SPH_C32(0xEC4E6C89)
#define CS8   SPH_C32(0x452821E6)
#define CS9   SPH_C32(0x38D01377)
#define CSA   SPH_C32(0xBE5466CF)
#define CSB   SPH_C32(0x34E90C6C)
#define CSC   SPH_C32(0xC0AC29B7)
#define CSD   SPH_C32(0xC97C50DD)
#define CSE   SPH_C32(0x3F84D5B5)
#define CSF   SPH_C32(0xB5470917)

#define GS(m0, m1, c0, c1, a, b, c, d)   do { \
        a = SPH_T32(a + b + (m0 ^ c1)); \
        d = SPH_ROTR32(d ^ a, 16); \
        c = SPH_T32(c + d); \
        b = SPH_ROTR32(b ^ c, 12); \
        a = SPH_T32(a + b + (m1 ^ c0)); \
        d = SPH_ROTR32(d ^ a, 8); \
        c = SPH_T32(c + d); \
        b = SPH_ROTR32(b ^ c, 7); \
    } while (0)

#define ROUND_S(r)   do { \
        GS(Mx(r, 0), Mx(r, 1), CSx(r, 0), CSx(r, 1), V0, V4, V8, VC); \
        GS(Mx(r, 2), Mx(r, 3), CSx(r, 2), CSx(r, 3), V1, V5, V9, VD); \
        GS(Mx(r, 4), Mx(r, 5), CSx(r, 4), CSx(r, 5), V2, V6, VA, VE); \
        GS(Mx(r, 6), Mx(r, 7), CSx(r, 6), CSx(r, 7), V3, V7, VB, VF); \
        GS(Mx(r, 8), Mx(r, 9), CSx(r, 8), CSx(r, 9), V0, V5, VA, VF); \
        GS(Mx(r, A), Mx(r, B), CSx(r, A), CSx(r, B), V1, V6, VB, VC); \
        GS(Mx(r, C), Mx(r, D), CSx(r, C), CSx(r, D), V2, V7, V8, VD); \
        GS(Mx(r, E), Mx(r, F), CSx(r, E), CSx(r, F), V3, V4, V9, VE); \
    } while (0)

#define COMPRESS32   do { \
        sph_u32 M0, M1, M2, M3, M4, M5, M6, M7; \
        sph_u32 M8, M9, MA, MB, MC, MD, ME, MF; \
        sph_u32 V0, V1, V2, V3, V4, V5, V6, V7; \
        sph_u32 V8, V9, VA, VB, VC, VD, VE, VF; \
        V0 = H0; \
        V1 = H1; \
        V2 = H2; \
        V3 = H3; \
        V4 = H4; \
        V5 = H5; \
        V6 = H6; \
        V7 = H7; \
        V8 = S0 ^ CS0; \
        V9 = S1 ^ CS1; \
        VA = S2 ^ CS2; \
        VB = S3 ^ CS3; \
        VC = T0 ^ CS4; \
        VD = T0 ^ CS5; \
        VE = T1 ^ CS6; \
        VF = T1 ^ CS7; \
        M0 = input[0]; \
        M1 = input[1]; \
        M2 = input[2]; \
        M3 = input[3]; \
        M4 = input[4]; \
        M5 = input[5]; \
        M6 = input[6]; \
        M7 = input[7]; \
        M8 = input[8]; \
        M9 = input[9]; \
        MA = input[10]; \
        MB = input[11]; \
        MC = input[12]; \
        MD = input[13]; \
        ME = input[14]; \
        MF = input[15]; \
        ROUND_S(0); \
        ROUND_S(1); \
        ROUND_S(2); \
        ROUND_S(3); \
        ROUND_S(4); \
        ROUND_S(5); \
        ROUND_S(6); \
        ROUND_S(7); \
        H0 ^= S0 ^ V0 ^ V8; \
        H1 ^= S1 ^ V1 ^ V9; \
        H2 ^= S2 ^ V2 ^ VA; \
        H3 ^= S3 ^ V3 ^ VB; \
        H4 ^= S0 ^ V4 ^ VC; \
        H5 ^= S1 ^ V5 ^ VD; \
        H6 ^= S2 ^ V6 ^ VE; \
        H7 ^= S3 ^ V7 ^ VF; \
    } while (0)


__global__ void kepler_blake256_hash( uint64_t *g_out, uint32_t nonce, uint32_t *g_good, bool validate )
{
    uint32_t input[16];
    uint64_t output[4];

#pragma unroll 16
    for (int i=0; i < 16; ++i) input[i] = pdata[i];

    sph_u32 H0 = 0x6A09E667;
    sph_u32 H1 = 0xBB67AE85;
    sph_u32 H2 = 0x3C6EF372;
    sph_u32 H3 = 0xA54FF53A;
    sph_u32 H4 = 0x510E527F;
    sph_u32 H5 = 0x9B05688C;
    sph_u32 H6 = 0x1F83D9AB;
    sph_u32 H7 = 0x5BE0CD19;
    sph_u32 S0 = 0;
    sph_u32 S1 = 0;
    sph_u32 S2 = 0;
    sph_u32 S3 = 0;
    sph_u32 T0 = 0;
    sph_u32 T1 = 0;
    T0 = SPH_T32(T0 + 512);
    COMPRESS32;

#pragma unroll 3
    for (int i=0; i < 3; ++i) input[i] = pdata[16+i];
    input[3] = nonce + ((blockIdx.x * blockDim.x) + threadIdx.x);
    input[4] = 0x80000000;
#pragma unroll 8
    for (int i=5; i < 13; ++i) input[i] = 0;
    input[13] = 0x00000001;
    input[14] = T1;
    input[15] = T0 + 128;

    T0 = SPH_T32(T0 + 128);
    COMPRESS32;

    cuda_sph_enc32be((unsigned char*)output + 4*6, H6);
    cuda_sph_enc32be((unsigned char*)output + 4*7, H7);
    if (validate || output[3] <=  ptarget64[3])
    {
        // this data is only needed when we actually need to save the hashes
        cuda_sph_enc32be((unsigned char*)output + 4*0, H0);
        cuda_sph_enc32be((unsigned char*)output + 4*1, H1);
        cuda_sph_enc32be((unsigned char*)output + 4*2, H2);
        cuda_sph_enc32be((unsigned char*)output + 4*3, H3);
        cuda_sph_enc32be((unsigned char*)output + 4*4, H4);
        cuda_sph_enc32be((unsigned char*)output + 4*5, H5);
    }

    if (validate)
    {
        g_out += 4 * ((blockIdx.x * blockDim.x) + threadIdx.x);
#pragma unroll 4
        for (int i=0; i < 4; ++i) g_out[i] = output[i];
    }

    if (output[3] <=  ptarget64[3]) {
        uint64_t *g_good64 = (uint64_t*)g_good;
        if (output[3] < g_good64[3]) {
            g_good64[3] = output[3];
            g_good64[2] = output[2];
            g_good64[1] = output[1];
            g_good64[0] = output[0];
            g_good[8] = nonce + ((blockIdx.x * blockDim.x) + threadIdx.x);
        }
    }
}

bool NVKernel::prepare_blake256(int thr_id, const uint32_t host_pdata[20], const uint32_t host_ptarget[8])
{
    static bool init[MAX_DEVICES] = {false};
    if (!init[thr_id])
    {
        // allocate pinned host memory for good hashes
        uint32_t *tmp;
        checkCudaErrors(cudaMalloc((void **) &tmp, 9*sizeof(uint32_t))); context_good[0][thr_id] = tmp;
        checkCudaErrors(cudaMalloc((void **) &tmp, 9*sizeof(uint32_t))); context_good[1][thr_id] = tmp;

        init[thr_id] = true;
    }
    checkCudaErrors(cudaMemcpyToSymbol(pdata, host_pdata, 20*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(ptarget64, host_ptarget, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

    return context_good[0][thr_id] && context_good[1][thr_id];
}

void NVKernel::do_blake256(dim3 grid, dim3 threads, int thr_id, int stream, uint32_t *hash, uint32_t nonce, int throughput, bool do_d2h)
{
    checkCudaErrors(cudaMemsetAsync(context_good[stream][thr_id], 0xff, 9 * sizeof(uint32_t), context_streams[stream][thr_id]));

    kepler_blake256_hash<<<grid, threads, 0, context_streams[stream][thr_id]>>>((uint64_t*)context_hash[stream][thr_id], nonce, context_good[stream][thr_id], do_d2h);

    // copy hashes from device memory to host (ALL hashes, lots of data...)
    if (do_d2h && hash != NULL) {
        size_t mem_size = throughput * sizeof(uint32_t) * 8;
        checkCudaErrors(cudaMemcpyAsync(hash, context_hash[stream][thr_id], mem_size,
                        cudaMemcpyDeviceToHost, context_streams[stream][thr_id]));
    }
    else if (hash != NULL) {
        // asynchronous copy of winning nonce (just 4 bytes...)
        checkCudaErrors(cudaMemcpyAsync(hash, context_good[stream][thr_id]+8, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, context_streams[stream][thr_id]));
    }
}
