//
//  =============== SHA256 part on nVidia GPU ======================
//
// NOTE: compile this .cu module for compute_10,sm_10 with --maxrregcount=64
//

#include <map>
#include <cuda.h>

#include "salsa_kernel.h"
#include "miner.h"

#include "sha256.h"

// define some error checking macros
#undef checkCudaErrors

#if WIN32
#define DELIMITER '/'
#else
#define DELIMITER '/'
#endif
#define __FILENAME__ ( strrchr(__FILE__, DELIMITER) != NULL ? strrchr(__FILE__, DELIMITER)+1 : __FILE__ )

#define checkCudaErrors(x) \
{ \
    cudaGetLastError(); \
    x; \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
        applog(LOG_ERR, "GPU #%d: cudaError %d (%s) calling '%s' (%s line %d)\n", device_map[thr_id], err, cudaGetErrorString(err), #x, __FILENAME__, __LINE__); \
}

// from salsa_kernel.cu
extern std::map<int, uint32_t *> context_idata[2];
extern std::map<int, uint32_t *> context_odata[2];
extern std::map<int, cudaStream_t> context_streams[2];
extern std::map<int, uint32_t *> context_tstate[2];
extern std::map<int, uint32_t *> context_ostate[2];
extern std::map<int, uint32_t *> context_hash[2];

static const uint32_t host_sha256_h[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

static const uint32_t host_sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

/* Elementary functions used by SHA256 */
#define Ch(x, y, z)     ((x & (y ^ z)) ^ z)
#define Maj(x, y, z)    ((x & (y | z)) | (y & z))
#define ROTR(x, n)      ((x >> n) | (x << (32 - n)))
#define S0(x)           (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define S1(x)           (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define s0(x)           (ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3))
#define s1(x)           (ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10))

/* SHA256 round function */
#define RND(a, b, c, d, e, f, g, h, k) \
    do { \
        t0 = h + S1(e) + Ch(e, f, g) + k; \
        t1 = S0(a) + Maj(a, b, c); \
        d += t0; \
        h  = t0 + t1; \
    } while (0)

/* Adjusted round function for rotating state */
#define RNDr(S, W, i) \
    RND(S[(64 - i) % 8], S[(65 - i) % 8], \
        S[(66 - i) % 8], S[(67 - i) % 8], \
        S[(68 - i) % 8], S[(69 - i) % 8], \
        S[(70 - i) % 8], S[(71 - i) % 8], \
        W[i] + sha256_k[i])

static const uint32_t host_keypad[12] = {
    0x80000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00000280
};

static const uint32_t host_innerpad[11] = {
    0x80000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x000004a0
};

static const uint32_t host_outerpad[8] = {
    0x80000000, 0, 0, 0, 0, 0, 0, 0x00000300
};

static const uint32_t host_finalblk[16] = {
    0x00000001, 0x80000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00000620
};

//
// CUDA code
//

__constant__ uint32_t sha256_h[8];
__constant__ uint32_t sha256_k[64];
__constant__ uint32_t keypad[12];
__constant__ uint32_t innerpad[11];
__constant__ uint32_t outerpad[8];
__constant__ uint32_t finalblk[16];
__constant__ uint32_t pdata[20];
__constant__ uint32_t midstate[8];

__device__ void mycpy12(uint32_t *d, const uint32_t *s) {
#pragma unroll 3
    for (int k=0; k < 3; k++) d[k] = s[k];
}

__device__ void mycpy16(uint32_t *d, const uint32_t *s) {
#pragma unroll 4
    for (int k=0; k < 4; k++) d[k] = s[k];
}

__device__ void mycpy32(uint32_t *d, const uint32_t *s) {
#pragma unroll 8
    for (int k=0; k < 8; k++) d[k] = s[k];
}

__device__ void mycpy44(uint32_t *d, const uint32_t *s) {
#pragma unroll 11
    for (int k=0; k < 11; k++) d[k] = s[k];
}

__device__ void mycpy48(uint32_t *d, const uint32_t *s) {
#pragma unroll 12
    for (int k=0; k < 12; k++) d[k] = s[k];
}

__device__ void mycpy64(uint32_t *d, const uint32_t *s) {
#pragma unroll 16
    for (int k=0; k < 16; k++) d[k] = s[k];
}

__device__ uint32_t cuda_swab32(uint32_t x)
{
    return (((x << 24) & 0xff000000u) | ((x << 8) & 0x00ff0000u)
          | ((x >> 8) & 0x0000ff00u) | ((x >> 24) & 0x000000ffu));
}

__device__ void mycpy32_swab32(uint32_t *d, const uint32_t *s) {
#pragma unroll 8
    for (int k=0; k < 8; k++) d[k] = cuda_swab32(s[k]);
}

__device__ void mycpy64_swab32(uint32_t *d, const uint32_t *s) {
#pragma unroll 16
    for (int k=0; k < 16; k++) d[k] = cuda_swab32(s[k]);
}

__device__ void cuda_sha256_init(uint32_t *state)
{
    mycpy32(state, sha256_h);
}

/*
 * SHA256 block compression function.  The 256-bit state is transformed via
 * the 512-bit input block to produce a new state. Modified for lower register use.
 */
__device__ void cuda_sha256_transform(uint32_t *state, const uint32_t *block)
{
    uint32_t W[64]; // only 4 of these are accessed during each partial Mix
    uint32_t S[8];
    uint32_t t0, t1;
    int i;

    /* 1. Initialize working variables. */
    mycpy32(S, state);

    /* 2. Prepare message schedule W and Mix. */
    mycpy16(W, block);
    RNDr(S, W,  0); RNDr(S, W,  1); RNDr(S, W,  2); RNDr(S, W,  3);

    mycpy16(W+4, block+4);
    RNDr(S, W,  4); RNDr(S, W,  5); RNDr(S, W,  6); RNDr(S, W,  7);

    mycpy16(W+8, block+8);
    RNDr(S, W,  8); RNDr(S, W,  9); RNDr(S, W, 10); RNDr(S, W, 11);

    mycpy16(W+12, block+12);
    RNDr(S, W, 12); RNDr(S, W, 13); RNDr(S, W, 14); RNDr(S, W, 15);

#pragma unroll 2
    for (i = 16; i < 20; i += 2) {
        W[i]   = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
        W[i+1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15]; }
    RNDr(S, W, 16); RNDr(S, W, 17); RNDr(S, W, 18); RNDr(S, W, 19);

#pragma unroll 2
    for (i = 20; i < 24; i += 2) {
        W[i]   = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
        W[i+1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15]; }
    RNDr(S, W, 20); RNDr(S, W, 21); RNDr(S, W, 22); RNDr(S, W, 23);

#pragma unroll 2
    for (i = 24; i < 28; i += 2) {
        W[i]   = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
        W[i+1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15]; }
    RNDr(S, W, 24); RNDr(S, W, 25); RNDr(S, W, 26); RNDr(S, W, 27);

#pragma unroll 2
    for (i = 28; i < 32; i += 2) {
        W[i]   = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
        W[i+1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15]; }
    RNDr(S, W, 28); RNDr(S, W, 29); RNDr(S, W, 30); RNDr(S, W, 31);

#pragma unroll 2
    for (i = 32; i < 36; i += 2) {
        W[i]   = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
        W[i+1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15]; }
    RNDr(S, W, 32); RNDr(S, W, 33); RNDr(S, W, 34); RNDr(S, W, 35);

#pragma unroll 2
    for (i = 36; i < 40; i += 2) {
        W[i]   = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
        W[i+1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15]; }
    RNDr(S, W, 36); RNDr(S, W, 37); RNDr(S, W, 38); RNDr(S, W, 39);

#pragma unroll 2
    for (i = 40; i < 44; i += 2) {
        W[i]   = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
        W[i+1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15]; }
    RNDr(S, W, 40); RNDr(S, W, 41); RNDr(S, W, 42); RNDr(S, W, 43);

#pragma unroll 2
    for (i = 44; i < 48; i += 2) {
        W[i]   = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
        W[i+1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15]; }
    RNDr(S, W, 44); RNDr(S, W, 45); RNDr(S, W, 46); RNDr(S, W, 47);

#pragma unroll 2
    for (i = 48; i < 52; i += 2) {
        W[i]   = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
        W[i+1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15]; }
    RNDr(S, W, 48); RNDr(S, W, 49); RNDr(S, W, 50); RNDr(S, W, 51);

#pragma unroll 2
    for (i = 52; i < 56; i += 2) {
        W[i]   = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
        W[i+1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15]; }
    RNDr(S, W, 52); RNDr(S, W, 53); RNDr(S, W, 54); RNDr(S, W, 55);

#pragma unroll 2
    for (i = 56; i < 60; i += 2) {
        W[i]   = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
        W[i+1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15]; }
    RNDr(S, W, 56); RNDr(S, W, 57); RNDr(S, W, 58); RNDr(S, W, 59);

#pragma unroll 2
    for (i = 60; i < 64; i += 2) {
        W[i]   = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
        W[i+1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15]; }
    RNDr(S, W, 60); RNDr(S, W, 61); RNDr(S, W, 62); RNDr(S, W, 63);

    /* 3. Mix local working variables into global state */
#pragma unroll 8
    for (i = 0; i < 8; i++)
        state[i] += S[i];
}

//
// HMAC SHA256 functions, modified to work with pdata and nonce directly
//

__device__ void cuda_HMAC_SHA256_80_init(uint32_t *tstate, uint32_t *ostate, uint32_t nonce)
{
    uint32_t ihash[8];
    uint32_t pad[16];
    int i;

    /* tstate is assumed to contain the midstate of key */
    mycpy12(pad, pdata + 16);
    pad[3] = nonce;
    mycpy48(pad + 4, keypad);
    cuda_sha256_transform(tstate, pad);
    mycpy32(ihash, tstate);

    cuda_sha256_init(ostate);
#pragma unroll 8
    for (i = 0; i < 8; i++)
        pad[i] = ihash[i] ^ 0x5c5c5c5c;
#pragma unroll 8
    for (i=8; i < 16; i++)
        pad[i] = 0x5c5c5c5c;
    cuda_sha256_transform(ostate, pad);

    cuda_sha256_init(tstate);
#pragma unroll 8
    for (i = 0; i < 8; i++)
        pad[i] = ihash[i] ^ 0x36363636;
#pragma unroll 8
    for (i=8; i < 16; i++)
        pad[i] = 0x36363636;
    cuda_sha256_transform(tstate, pad);
}

__device__ void cuda_PBKDF2_SHA256_80_128(const uint32_t *tstate,
    const uint32_t *ostate, uint32_t *output, uint32_t nonce)
{
    uint32_t istate[8], ostate2[8];
    uint32_t ibuf[16], obuf[16];

    mycpy32(istate, tstate);
    cuda_sha256_transform(istate, pdata);
    
    mycpy12(ibuf, pdata + 16);
    ibuf[3] = nonce;
    ibuf[4] = 1;
    mycpy44(ibuf + 5, innerpad);

    mycpy32(obuf, istate);
    mycpy32(obuf + 8, outerpad);
    cuda_sha256_transform(obuf, ibuf);

    mycpy32(ostate2, ostate);
    cuda_sha256_transform(ostate2, obuf);
    mycpy32_swab32(output, ostate2);       // TODO: coalescing would be desired

    mycpy32(obuf, istate);
    ibuf[4] = 2;
    cuda_sha256_transform(obuf, ibuf);

    mycpy32(ostate2, ostate);
    cuda_sha256_transform(ostate2, obuf);
    mycpy32_swab32(output+8, ostate2);     // TODO: coalescing would be desired

    mycpy32(obuf, istate);
    ibuf[4] = 3;
    cuda_sha256_transform(obuf, ibuf);

    mycpy32(ostate2, ostate);
    cuda_sha256_transform(ostate2, obuf);
    mycpy32_swab32(output+16, ostate2);    // TODO: coalescing would be desired

    mycpy32(obuf, istate);
    ibuf[4] = 4;
    cuda_sha256_transform(obuf, ibuf);

    mycpy32(ostate2, ostate);
    cuda_sha256_transform(ostate2, obuf);
    mycpy32_swab32(output+24, ostate2);    // TODO: coalescing would be desired
}

__global__ void cuda_pre_sha256(uint32_t g_inp[32], uint32_t g_tstate_ext[8], uint32_t g_ostate_ext[8], uint32_t nonce)
{
    nonce        +=       (blockIdx.x * blockDim.x) + threadIdx.x; 
    g_inp        += 32 * ((blockIdx.x * blockDim.x) + threadIdx.x);
    g_tstate_ext +=  8 * ((blockIdx.x * blockDim.x) + threadIdx.x);
    g_ostate_ext +=  8 * ((blockIdx.x * blockDim.x) + threadIdx.x);

    uint32_t tstate[8], ostate[8];
    mycpy32(tstate, midstate);

    cuda_HMAC_SHA256_80_init(tstate, ostate, nonce);

    mycpy32(g_tstate_ext, tstate);            // TODO: coalescing would be desired
    mycpy32(g_ostate_ext, ostate);            // TODO: coalescing would be desired

    cuda_PBKDF2_SHA256_80_128(tstate, ostate, g_inp, nonce);
}

__global__ void cuda_post_sha256(uint32_t g_output[8], uint32_t g_tstate_ext[8], uint32_t g_ostate_ext[8], uint32_t g_salt_ext[32])
{
    g_output     +=  8 * ((blockIdx.x * blockDim.x) + threadIdx.x);
    g_tstate_ext +=  8 * ((blockIdx.x * blockDim.x) + threadIdx.x);
    g_ostate_ext +=  8 * ((blockIdx.x * blockDim.x) + threadIdx.x);
    g_salt_ext   += 32 * ((blockIdx.x * blockDim.x) + threadIdx.x);

    uint32_t tstate[16];
    mycpy32(tstate, g_tstate_ext);            // TODO: coalescing would be desired
    
    uint32_t halfsalt[16];
    mycpy64_swab32(halfsalt, g_salt_ext);     // TODO: coalescing would be desired
    cuda_sha256_transform(tstate, halfsalt);
    mycpy64_swab32(halfsalt, g_salt_ext+16);  // TODO: coalescing would be desired
    cuda_sha256_transform(tstate, halfsalt);
    cuda_sha256_transform(tstate, finalblk);

    uint32_t buf[16];
    mycpy32(buf, tstate);
    mycpy32(buf + 8, outerpad);

    uint32_t ostate[16];
    mycpy32(ostate, g_ostate_ext);

    cuda_sha256_transform(ostate, buf);
    mycpy32_swab32(g_output, ostate);        // TODO: coalescing would be desired
}

//
// callable host code to initialize constants and to call kernels
//

extern "C" void prepare_sha256(int thr_id, uint32_t host_pdata[20], uint32_t host_midstate[8])
{
    static bool init[8] = {false, false, false, false, false, false, false, false};
    if (!init[thr_id])
    {
        checkCudaErrors(cudaMemcpyToSymbol(sha256_h, host_sha256_h, sizeof(host_sha256_h), 0, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(sha256_k, host_sha256_k, sizeof(host_sha256_k), 0, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(keypad, host_keypad, sizeof(host_keypad), 0, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(innerpad, host_innerpad, sizeof(host_innerpad), 0, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(outerpad, host_outerpad, sizeof(host_outerpad), 0, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(finalblk, host_finalblk, sizeof(host_finalblk), 0, cudaMemcpyHostToDevice));
        init[thr_id] = true;
    }
    checkCudaErrors(cudaMemcpyToSymbol(pdata, host_pdata, 20*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(midstate, host_midstate, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}

extern "C" void pre_sha256(int thr_id, int stream, uint32_t nonce, int throughput)
{
    dim3 block(128);
    dim3 grid((throughput+127)/128);

    cuda_pre_sha256<<<grid, block, 0, context_streams[stream][thr_id]>>>(context_idata[stream][thr_id], context_tstate[stream][thr_id], context_ostate[stream][thr_id], nonce);
}

extern "C" void post_sha256(int thr_id, int stream, int throughput)
{
    dim3 block(128);
    dim3 grid((throughput+127)/128);

    cuda_post_sha256<<<grid, block, 0, context_streams[stream][thr_id]>>>(context_hash[stream][thr_id], context_tstate[stream][thr_id], context_ostate[stream][thr_id], context_odata[stream][thr_id]);
}
