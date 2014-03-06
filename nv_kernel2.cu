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
#include "nv_kernel2.h"

#define THREADS_PER_WU 1  // single thread per hash 

#if __CUDA_ARCH__ < 350 
    // Kepler (Compute 3.0)
    #define __ldg(x) (*(x))
#endif

// grab lane ID
static __device__ __inline__ unsigned int __laneId() { unsigned int laneId; asm( "mov.u32 %0, %%laneid;" : "=r"( laneId ) ); return laneId; }

// forward references
template <int ALGO> __global__ void nv2_scrypt_core_kernelA(uint32_t *g_idata, int begin, int end);
template <int ALGO> __global__ void nv2_scrypt_core_kernelB(uint32_t *g_odata, int begin, int end);
template <int ALGO> __global__ void nv2_scrypt_core_kernelA_LG(uint32_t *g_idata, int begin, int end, unsigned int LOOKUP_GAP);
template <int ALGO> __global__ void nv2_scrypt_core_kernelB_LG(uint32_t *g_odata, int begin, int end, unsigned int LOOKUP_GAP);

// scratchbuf constants (pointers to scratch buffer for each work unit)
__constant__ uint32_t* c_V[TOTAL_WARP_LIMIT];

// iteration count N
__constant__ uint32_t c_N;
__constant__ uint32_t c_N_1; // N - 1
__constant__ uint32_t c_spacing; // (N+LOOKUP_GAP-1)/LOOKUP_GAP


NV2Kernel::NV2Kernel() : KernelInterface()
{
}

void NV2Kernel::set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_V, h_V, MAXWARPS*sizeof(uint32_t*), 0, cudaMemcpyHostToDevice));
}

bool NV2Kernel::run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, unsigned int N, unsigned int LOOKUP_GAP, bool interactive, bool benchmark, int texture_cache)
{
    bool success = true;

    // clear CUDA's error variable
    cudaGetLastError();

    // make some constants available to kernel, update only initially and when changing
    static int prev_N[8] = {0,0,0,0,0,0,0,0};
    if (N != prev_N[thr_id]) {
        uint32_t h_N = N;
        checkCudaErrors(cudaMemcpyToSymbolAsync(c_N, &h_N, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream));
        uint32_t h_N_1 = N-1;
        checkCudaErrors(cudaMemcpyToSymbolAsync(c_N_1, &h_N_1, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream));
        uint32_t h_spacing = (N+LOOKUP_GAP-1)/LOOKUP_GAP;
        checkCudaErrors(cudaMemcpyToSymbolAsync(c_spacing, &h_spacing, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream));
        prev_N[thr_id] = N;
    }

    // First phase: Sequential writes to scratchpad.
    const int batch = device_batchsize[thr_id];
    const int sleeptime = 100;
    unsigned int pos = 0;
    int situation = 0;

    do 
    {
        if (LOOKUP_GAP == 1)
            switch(opt_algo) {
                case ALGO_SCRYPT:      nv2_scrypt_core_kernelA<ALGO_SCRYPT>     <<< grid, threads, 0, stream >>>(d_idata, pos, min(pos+batch, N)); break;
                case ALGO_SCRYPT_JANE: nv2_scrypt_core_kernelA<ALGO_SCRYPT_JANE><<< grid, threads, 0, stream >>>(d_idata, pos, min(pos+batch, N)); break;
            }
        else
            switch(opt_algo) {
                case ALGO_SCRYPT:      nv2_scrypt_core_kernelA_LG<ALGO_SCRYPT>     <<< grid, threads, 0, stream >>>(d_idata, pos, min(pos+batch, N), LOOKUP_GAP); break;
                case ALGO_SCRYPT_JANE: nv2_scrypt_core_kernelA_LG<ALGO_SCRYPT_JANE><<< grid, threads, 0, stream >>>(d_idata, pos, min(pos+batch, N), LOOKUP_GAP); break;
            }
        
        if (!benchmark && interactive) {
            checkCudaErrors(MyStreamSynchronize(stream, ++situation, thr_id));
            usleep(sleeptime);
        }

        pos += batch;
    } while (pos < N);

    // Second phase: Random read access from scratchpad.
    pos = 0;
    do
    {
        if (pos > 0 && !benchmark && interactive) {
            checkCudaErrors(MyStreamSynchronize(stream, ++situation, thr_id));
            usleep(sleeptime);
        }

        if (LOOKUP_GAP == 1)
            switch(opt_algo) {
                case ALGO_SCRYPT:      nv2_scrypt_core_kernelB<ALGO_SCRYPT     ><<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N)); break;
                case ALGO_SCRYPT_JANE: nv2_scrypt_core_kernelB<ALGO_SCRYPT_JANE><<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N)); break;
            }
        else
            switch(opt_algo) {
                case ALGO_SCRYPT:      nv2_scrypt_core_kernelB_LG<ALGO_SCRYPT     ><<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N), LOOKUP_GAP); break;
                case ALGO_SCRYPT_JANE: nv2_scrypt_core_kernelB_LG<ALGO_SCRYPT_JANE><<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N), LOOKUP_GAP); break;
            }

        pos += batch;
    } while (pos < N);

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

__device__ __forceinline__ void __transposed_read_BC(const uint4 *S, uint4 (&B)[4], uint4 (&C)[4], int spacing, int row)
{
    unsigned int laneId = __laneId();

    unsigned int lane8 = laneId%8;
    unsigned int tile  = laneId/8;

    // Perform the same transposition as in __transposed_write_BC, but in reverse order.
    // See the illustrations in comments for __transposed_write_BC.

    // read and rotate rows, in reverse row order
    uint4 T1[8], T2[8];
    T1[7] = __ldg(&S[(spacing*2*(32*tile   ) +  lane8      + 8*__shfl(row, 0, 8))]);
    T1[6] = __ldg(&S[(spacing*2*(32*tile+4 ) + (lane8+7)%8 + 8*__shfl(row, 1, 8))]);
    T1[5] = __ldg(&S[(spacing*2*(32*tile+8 ) + (lane8+6)%8 + 8*__shfl(row, 2, 8))]);
    T1[4] = __ldg(&S[(spacing*2*(32*tile+12) + (lane8+5)%8 + 8*__shfl(row, 3, 8))]);
    T1[3] = __ldg(&S[(spacing*2*(32*tile+16) + (lane8+4)%8 + 8*__shfl(row, 4, 8))]);
    T1[2] = __ldg(&S[(spacing*2*(32*tile+20) + (lane8+3)%8 + 8*__shfl(row, 5, 8))]);
    T1[1] = __ldg(&S[(spacing*2*(32*tile+24) + (lane8+2)%8 + 8*__shfl(row, 6, 8))]);
    T1[0] = __ldg(&S[(spacing*2*(32*tile+28) + (lane8+1)%8 + 8*__shfl(row, 7, 8))]);
    
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

__device__ __forceinline__ void __transposed_xor_BC(const uint4 *S, uint4 (&B)[4], uint4 (&C)[4], int spacing, int row)
{
    uint4 BT[4], CT[4];
    __transposed_read_BC(S, BT, CT, spacing, row);

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
//! Experimental Scrypt core kernel for Titan devices.
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
template <int ALGO> __global__ void nv2_scrypt_core_kernelA(uint32_t *g_idata, int begin, int end)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x / warpSize * warpSize;
    g_idata += 32 * offset;
    uint32_t * V = c_V[offset / warpSize];
    uint4 B[4], C[4];
    int i = begin;

    if(i == 0) {
        __transposed_read_BC((uint4*)g_idata, B, C, 1, 0);
        __transposed_write_BC(B, C, (uint4*)V, c_N); 
        ++i;
    } else
        __transposed_read_BC((uint4*)(V + (i-1)*32), B, C, c_N, 0);

    while(i < end) {
        block_mixer<ALGO>(B, C); block_mixer<ALGO>(C, B);
        __transposed_write_BC(B, C, (uint4*)(V + i*32), c_N); 
        ++i;
    }
}

template <int ALGO> __global__ void nv2_scrypt_core_kernelA_LG(uint32_t *g_idata, int begin, int end, unsigned int LOOKUP_GAP)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x / warpSize * warpSize;
    g_idata += 32 * offset;
    uint32_t * V = c_V[offset / warpSize];
    uint4 B[4], C[4];
    int i = begin;

    if(i == 0) {
        __transposed_read_BC((uint4*)g_idata, B, C, 1, 0);
        __transposed_write_BC(B, C, (uint4*)V, c_spacing); 
        ++i;
    } else {
        int pos = (i-1)/LOOKUP_GAP, loop = (i-1)-pos*LOOKUP_GAP;
        __transposed_read_BC((uint4*)(V + pos*32), B, C, c_spacing, 0);
        while(loop--) { block_mixer<ALGO>(B, C); block_mixer<ALGO>(C, B); }
    }

    while(i < end) {
        block_mixer<ALGO>(B, C); block_mixer<ALGO>(C, B);
        if (i % LOOKUP_GAP == 0)
          __transposed_write_BC(B, C, (uint4*)(V + (i/LOOKUP_GAP)*32), c_spacing); 
        ++i;
    }
}

template <int ALGO> __global__ void nv2_scrypt_core_kernelB(uint32_t *g_odata, int begin, int end)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x / warpSize * warpSize;
    g_odata += 32 * offset;
    uint32_t * V = c_V[offset / warpSize];
    uint4 B[4], C[4];

    if(begin == 0) {
        __transposed_read_BC((uint4*)V, B, C, c_N, c_N_1);
        block_mixer<ALGO>(B, C); block_mixer<ALGO>(C, B);
    } else
        __transposed_read_BC((uint4*)g_odata, B, C, 1, 0);

    for (int i = begin; i < end; i++)  {
        int slot = C[0].x & c_N_1;
        __transposed_xor_BC((uint4*)(V), B, C, c_N, slot);
        block_mixer<ALGO>(B, C); block_mixer<ALGO>(C, B);
    }

    __transposed_write_BC(B, C, (uint4*)(g_odata), 1);
}

template <int ALGO> __global__ void nv2_scrypt_core_kernelB_LG(uint32_t *g_odata, int begin, int end, unsigned int LOOKUP_GAP)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x / warpSize * warpSize;
    g_odata += 32 * offset;
    uint32_t * V = c_V[offset / warpSize];
    uint4 B[4], C[4];

    if(begin == 0) {
      int pos = c_N_1/LOOKUP_GAP, loop = 1 + (c_N_1-pos*LOOKUP_GAP);
      __transposed_read_BC((uint4*)V, B, C, c_spacing, pos);
      while(loop--) { block_mixer<ALGO>(B, C); block_mixer<ALGO>(C, B); }
    } else {
        __transposed_read_BC((uint4*)g_odata, B, C, 1, 0);
    }

    for (int i = begin; i < end; i++)  {
        int slot = C[0].x & c_N_1;
        int pos = slot/LOOKUP_GAP, loop = slot-pos*LOOKUP_GAP;
        uint4 b[4], c[4]; __transposed_read_BC((uint4*)(V), b, c, c_spacing, pos);
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

// ROL macro replaced with the inline assembly code below to work around a performance issue
//#define ROL(a, offset) ((((uint64_t)a) << ((offset) % 64)) ^ (((uint64_t)a) >> (64-((offset) % 64))))
__inline__ __device__ uint2 ROL(const uint2 a, const int offset) {
    uint2 result;
    if(offset >= 32) {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
    } else {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
    }
    return result;
}
#define ROL_mult8(a, offset) ROL(a, offset)

__inline__ __device__ uint64_t devectorize(uint2 v) { return __double_as_longlong(__hiloint2double(v.y, v.x)); }
__inline__ __device__ uint2 vectorize(uint64_t v) { return make_uint2(__double2loint(__longlong_as_double(v)), __double2hiint(__longlong_as_double(v))); }
__inline__ __device__ uint2 operator^ (uint2 a, uint2 b) { return make_uint2(a.x ^ b.x, a.y ^ b.y); }
__inline__ __device__ uint2 operator& (uint2 a, uint2 b) { return make_uint2(a.x & b.x, a.y & b.y); }
__inline__ __device__ uint2 operator| (uint2 a, uint2 b) { return make_uint2(a.x | b.x, a.y | b.y); }
__inline__ __device__ uint2 operator~ (uint2 a) { return make_uint2(~a.x, ~a.y); }
__inline__ __device__ void operator^= (uint2 &a, uint2 b) { a = a ^ b; }

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

// in this implementation the first and last iteration of the for() loop were explicitly
// unrolled and redundant operations were removed (e.g. operations on zero inputs, and
// computation of unnecessary outputs)
__global__ void titan_crypto_hash( uint64_t *g_out, uint32_t nonce, uint32_t *g_good, bool validate )
{
    uint2 Aba, Abe, Abi, Abo, Abu;
    uint2 Aga, Age, Agi, Ago, Agu;
    uint2 Aka, Ake, Aki, Ako, Aku;
    uint2 Ama, Ame, Ami, Amo, Amu;
    uint2 Asa, Ase, Asi, Aso, Asu;
    uint2 BCa, BCe, BCi, BCo, BCu;
    uint2 Da, De, Di, Do, Du;
    uint2 Eba, Ebe, Ebi, Ebo, Ebu;
    uint2 Ega, Ege, Egi, Ego, Egu;
    uint2 Eka, Eke, Eki, Eko, Eku;
    uint2 Ema, Eme, Emi, Emo, Emu;
    uint2 Esa, Ese, Esi, Eso, Esu;

    // embed unique nonce into source data stream in pdata[]
    Agu = vectorize((pdata64[9] & 0x00000000FFFFFFFFULL) | (((uint64_t)cuda_swab32(nonce + ((blockIdx.x * blockDim.x) + threadIdx.x))) << 32));

    //    prepareTheta
    BCa = vectorize(pdata64[0]^pdata64[5]^0x0000000000000001ULL);
    BCe = vectorize(pdata64[1]^pdata64[6]^0x8000000000000000ULL);
    BCi = vectorize(pdata64[2]^pdata64[7]);
    BCo = vectorize(pdata64[3]^pdata64[8]);
    BCu = vectorize(pdata64[4])^Agu;

    //thetaRhoPiChiIotaPrepareTheta(round  , A, E)
    Da = BCu^ROL(BCe, 1);
    De = BCa^ROL(BCi, 1);
    Di = BCe^ROL(BCo, 1);
    Do = BCi^ROL(BCu, 1);
    Du = BCo^ROL(BCa, 1);

    Aba = vectorize(pdata64[0]) ^ Da;
    BCa = Aba;
    Age = vectorize(pdata64[6]) ^ De;
    BCe = ROL(Age, 44);
    Aki = Di;
    BCi = ROL(Aki, 43);
    Amo = Do;
    BCo = ROL(Amo, 21);
    Asu = Du;
    BCu = ROL(Asu, 14);
    Eba =   BCa ^((~BCe)&  BCi );
    Eba ^= vectorize((uint64_t)KeccakF_RoundConstants[0]);
    Ebe =   BCe ^((~BCi)&  BCo );
    Ebi =   BCi ^((~BCo)&  BCu );
    Ebo =   BCo ^((~BCu)&  BCa );
    Ebu =   BCu ^((~BCa)&  BCe );

    Abo = vectorize(pdata64[3]) ^ Do;
    BCa = ROL(Abo, 28);
    Agu ^= Du;
    BCe = ROL(Agu, 20);
    Aka = vectorize(0x0000000000000001ULL) ^ Da;
    BCi = ROL(Aka,  3);
    Ame = vectorize(0x8000000000000000ULL) ^ De;
    BCo = ROL(Ame, 45);
    Asi = Di;
    BCu = ROL(Asi, 61);
    Ega =   BCa ^((~BCe)&  BCi );
    Ege =   BCe ^((~BCi)&  BCo );
    Egi =   BCi ^((~BCo)&  BCu );
    Ego =   BCo ^((~BCu)&  BCa );
    Egu =   BCu ^((~BCa)&  BCe );

    Abe = vectorize(pdata64[1]) ^ De;
    BCa = ROL(Abe,  1);
    Agi = vectorize(pdata64[7]) ^ Di;
    BCe = ROL(Agi,  6);
    Ako = Do;
    BCi = ROL(Ako, 25);
    Amu = Du;
    BCo = ROL(Amu,  8);
    Asa = Da;
    BCu = ROL(Asa, 18);
    Eka =   BCa ^((~BCe)&  BCi );
    Eke =   BCe ^((~BCi)&  BCo );
    Eki =   BCi ^((~BCo)&  BCu );
    Eko =   BCo ^((~BCu)&  BCa );
    Eku =   BCu ^((~BCa)&  BCe );

    Abu = vectorize(pdata64[4]) ^ Du;
    BCa = ROL(Abu, 27);
    Aga = vectorize(pdata64[5]) ^ Da;
    BCe = ROL(Aga, 36);
    Ake = De;
    BCi = ROL(Ake, 10);
    Ami = Di;
    BCo = ROL(Ami, 15);
    Aso = Do;
    BCu = ROL(Aso, 56);
    Ema =   BCa ^((~BCe)&  BCi );
    Eme =   BCe ^((~BCi)&  BCo );
    Emi =   BCi ^((~BCo)&  BCu );
    Emo =   BCo ^((~BCu)&  BCa );
    Emu =   BCu ^((~BCa)&  BCe );

    Abi = vectorize(pdata64[2]) ^ Di;
    BCa = ROL(Abi, 62);
    Ago = vectorize(pdata64[8]) ^ Do;
    BCe = ROL(Ago, 55);
    Aku = Du;
    BCi = ROL(Aku, 39);
    Ama = Da;
    BCo = ROL(Ama, 41);
    Ase = De;
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
    Aba ^= vectorize((uint64_t)KeccakF_RoundConstants[1]);
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
    BCo = ROL(Emu, 8);
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
    BCu = ROL(Eso, 56);
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

//#pragma unroll 10
    for( int laneCount = 2; laneCount < 22; laneCount += 2 )
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
        Eba ^= vectorize((uint64_t)KeccakF_RoundConstants[laneCount]);
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
        BCo = ROL(Amu,  8);
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
        BCu = ROL(Aso, 56);
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
        Aba ^= vectorize((uint64_t)KeccakF_RoundConstants[laneCount+1]);
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
        BCo = ROL(Emu, 8);
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
        BCu = ROL(Eso, 56);
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
    Eba ^= vectorize((uint64_t)KeccakF_RoundConstants[22]);
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
    BCo = ROL(Amu,  8);
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
    BCu = ROL(Aso, 56);
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
    Aba ^= vectorize((uint64_t)KeccakF_RoundConstants[23]);
    Abe =   BCe ^((~BCi)&  BCo );
    Abi =   BCi ^((~BCo)&  BCu );
    Abo =   BCo ^((~BCu)&  BCa );

    if (validate) {
        g_out += 4 * ((blockIdx.x * blockDim.x) + threadIdx.x);
        g_out[3] = devectorize(Abo);
        g_out[2] = devectorize(Abi);
        g_out[1] = devectorize(Abe);
        g_out[0] = devectorize(Aba);
    }
    
    // the likelyhood of meeting the hashing target is so low, that we're not guarding this
    // with atomic writes, locks or similar...
    uint64_t *g_good64 = (uint64_t*)g_good;
    if (devectorize(Abo) <=  ptarget64[3]) {
        if (devectorize(Abo) < g_good64[3]) {
            g_good64[3] = devectorize(Abo);
            g_good64[2] = devectorize(Abi);
            g_good64[1] = devectorize(Abe);
            g_good64[0] = devectorize(Aba);
            g_good[8] = nonce + ((blockIdx.x * blockDim.x) + threadIdx.x);
        }
    }
}

static std::map<int, uint32_t *> context_good[2];

void NV2Kernel::prepare_keccak256(int thr_id, const uint32_t host_pdata[20], const uint32_t host_ptarget[8])
{
    static bool init[8] = {false, false, false, false, false, false, false, false};
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
}

bool NV2Kernel::do_keccak256(dim3 grid, dim3 threads, int thr_id, int stream, uint32_t *hash, uint32_t nonce, int throughput, bool do_d2h)
{
    bool success = true;
  
    checkCudaErrors(cudaMemsetAsync(context_good[stream][thr_id], 0xff, 9 * sizeof(uint32_t), context_streams[stream][thr_id]));

    titan_crypto_hash<<<grid, threads, 0, context_streams[stream][thr_id]>>>((uint64_t*)context_hash[stream][thr_id], nonce, context_good[stream][thr_id], do_d2h);

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

        // catch any kernel launch failures
    if (cudaPeekAtLastError() != cudaSuccess) success = false;

    return success;
}
