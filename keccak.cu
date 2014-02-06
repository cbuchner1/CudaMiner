//
//  =============== KECCAK part on nVidia GPU ======================
//
// NOTE: compile this .cu module for compute_10,sm_10 with --maxrregcount=64
//
// TODO: the actual CUDA porting work is work in progress...
//
//       For good performance we have to get rid of most local memory spills
//       TODO: make sure all loops have known trip counts at compile time
//             and are adequately unrolled.

#include <map>
#include <stdint.h>

#include "salsa_kernel.h"
#include "miner.h"

#include "keccak.h"

// define some error checking macros
#undef checkCudaErrors

#if WIN32
#define DELIMITER '\\'
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
extern std::map<int, uint32_t *> context_hash[2];

#define ROTL64(a,b) (((a) << (b)) | ((a) >> (64 - b)))

// CB
#define U32TO64_LE(p) \
    (((uint64_t)(*p)) | (((uint64_t)(*(p + 1))) << 32))

#define U64TO32_LE(p, v) \
    *p = (uint32_t)((v)); *(p+1) = (uint32_t)((v) >> 32);

static __device__ void mycpy64(uint32_t *d, const uint32_t *s) {
#pragma unroll 16
    for (int k=0; k < 16; ++k) d[k] = s[k];
}

static __device__ void mycpy56(uint32_t *d, const uint32_t *s) {
#pragma unroll 14
    for (int k=0; k < 14; ++k) d[k] = s[k];
}

static __device__ void mycpy32(uint32_t *d, const uint32_t *s) {
#pragma unroll 8
    for (int k=0; k < 8; ++k) d[k] = s[k];
}

static __device__ void mycpy8(uint32_t *d, const uint32_t *s) {
#pragma unroll 2
    for (int k=0; k < 2; ++k) d[k] = s[k];
}

static __device__ void mycpy4(uint32_t *d, const uint32_t *s) {
    *d = *s;
}

// ---------------------------- BEGIN keccak functions ------------------------------------

#define KECCAK_HASH "Keccak-512"

typedef struct keccak_hash_state_t {
    uint64_t state[25];                        // 25*2
    uint32_t buffer[72/4];                     // 72
} keccak_hash_state;

__device__ void statecopy0(keccak_hash_state *d, keccak_hash_state *s)
{
#pragma unroll 25
    for (int i=0; i < 25; ++i)
        d->state[i] = s->state[i];
}

__device__ void statecopy8(keccak_hash_state *d, keccak_hash_state *s)
{
#pragma unroll 25
    for (int i=0; i < 25; ++i)
        d->state[i] = s->state[i];
#pragma unroll 2
    for (int i=0; i < 2; ++i)
        d->buffer[i] = s->buffer[i];
}

static const uint64_t host_keccak_round_constants[24] = {
    0x0000000000000001ull, 0x0000000000008082ull,
    0x800000000000808aull, 0x8000000080008000ull,
    0x000000000000808bull, 0x0000000080000001ull,
    0x8000000080008081ull, 0x8000000000008009ull,
    0x000000000000008aull, 0x0000000000000088ull,
    0x0000000080008009ull, 0x000000008000000aull,
    0x000000008000808bull, 0x800000000000008bull,
    0x8000000000008089ull, 0x8000000000008003ull,
    0x8000000000008002ull, 0x8000000000000080ull,
    0x000000000000800aull, 0x800000008000000aull,
    0x8000000080008081ull, 0x8000000000008080ull,
    0x0000000080000001ull, 0x8000000080008008ull
};

__constant__ uint64_t c_keccak_round_constants[24];
__constant__ uint32_t pdata[20];

__device__ void
keccak_block(keccak_hash_state *S, const uint32_t *in) {
    size_t i;
    uint64_t *s = S->state, t[5], u[5], v, w;

    /* absorb input */
#pragma unroll 9
    for (i = 0; i < 72 / 8; i++, in += 2)
        s[i] ^= U32TO64_LE(in);
    
    for (i = 0; i < 24; i++) {
        /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
        t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
        t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
        t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
        t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
        t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

        /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
        u[0] = t[4] ^ ROTL64(t[1], 1);
        u[1] = t[0] ^ ROTL64(t[2], 1);
        u[2] = t[1] ^ ROTL64(t[3], 1);
        u[3] = t[2] ^ ROTL64(t[4], 1);
        u[4] = t[3] ^ ROTL64(t[0], 1);

        /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
        s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
        s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
        s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
        s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
        s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

        /* rho pi: b[..] = rotl(a[..], ..) */
        v = s[ 1];
        s[ 1] = ROTL64(s[ 6], 44);
        s[ 6] = ROTL64(s[ 9], 20);
        s[ 9] = ROTL64(s[22], 61);
        s[22] = ROTL64(s[14], 39);
        s[14] = ROTL64(s[20], 18);
        s[20] = ROTL64(s[ 2], 62);
        s[ 2] = ROTL64(s[12], 43);
        s[12] = ROTL64(s[13], 25);
        s[13] = ROTL64(s[19],  8);
        s[19] = ROTL64(s[23], 56);
        s[23] = ROTL64(s[15], 41);
        s[15] = ROTL64(s[ 4], 27);
        s[ 4] = ROTL64(s[24], 14);
        s[24] = ROTL64(s[21],  2);
        s[21] = ROTL64(s[ 8], 55);
        s[ 8] = ROTL64(s[16], 45);
        s[16] = ROTL64(s[ 5], 36);
        s[ 5] = ROTL64(s[ 3], 28);
        s[ 3] = ROTL64(s[18], 21);
        s[18] = ROTL64(s[17], 15);
        s[17] = ROTL64(s[11], 10);
        s[11] = ROTL64(s[ 7],  6);
        s[ 7] = ROTL64(s[10],  3);
        s[10] = ROTL64(    v,  1);

        /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
        v = s[ 0]; w = s[ 1]; s[ 0] ^= (~w) & s[ 2]; s[ 1] ^= (~s[ 2]) & s[ 3]; s[ 2] ^= (~s[ 3]) & s[ 4]; s[ 3] ^= (~s[ 4]) & v; s[ 4] ^= (~v) & w;
        v = s[ 5]; w = s[ 6]; s[ 5] ^= (~w) & s[ 7]; s[ 6] ^= (~s[ 7]) & s[ 8]; s[ 7] ^= (~s[ 8]) & s[ 9]; s[ 8] ^= (~s[ 9]) & v; s[ 9] ^= (~v) & w;
        v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
        v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
        v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

        /* iota: a[0,0] ^= round constant */
        s[0] ^= c_keccak_round_constants[i];
    }
}

__device__ void
keccak_hash_init(keccak_hash_state *S) { 
#pragma unroll 25
    for (int i=0; i<25; ++i)
        S->state[i] = 0ULL;
}

// assuming there is no leftover data and exactly 72 bytes are incoming
// we can directly call into the block hashing function
__device__ void
keccak_hash_update72(keccak_hash_state *S, const uint32_t *in) {
    keccak_block(S, in);
}

__device__ void keccak_hash_update8(keccak_hash_state *S, const uint32_t *in) {
    mycpy8(S->buffer, in);
}

__device__ void keccak_hash_update4_8(keccak_hash_state *S, const uint32_t *in) {
    mycpy4(S->buffer+8/4, in);
}

__device__ void keccak_hash_update4_56(keccak_hash_state *S, const uint32_t *in) {
    mycpy4(S->buffer+56/4, in);
}

__device__ void keccak_hash_update56(keccak_hash_state *S, const uint32_t *in) {
    mycpy56(S->buffer, in);
}

__device__ void keccak_hash_update64(keccak_hash_state *S, const uint32_t *in) {
    mycpy64(S->buffer, in);
}

__device__ void
keccak_hash_finish8(keccak_hash_state *S, uint32_t *hash) {
    S->buffer[8/4] = 0x01;
#pragma unroll 15
    for (int i=8/4+1; i < 72/4; ++i) S->buffer[i] = 0;
    S->buffer[72/4 - 1] |= 0x80000000;
    keccak_block(S, (const uint32_t*)S->buffer);
#pragma unroll 8
    for (size_t i = 0; i < 64; i += 8) {
        U64TO32_LE((&hash[i/4]), S->state[i / 8]);
    }
}

__device__ void
keccak_hash_finish12(keccak_hash_state *S, uint32_t *hash) {
    S->buffer[12/4] = 0x01;
#pragma unroll 14
    for (int i=12/4+1; i < 72/4; ++i) S->buffer[i] = 0;
    S->buffer[72/4 - 1] |= 0x80000000;
    keccak_block(S, (const uint32_t*)S->buffer);
#pragma unroll 8
    for (size_t i = 0; i < 64; i += 8) {
        U64TO32_LE((&hash[i/4]), S->state[i / 8]);
    }
}

__device__ void
keccak_hash_finish60(keccak_hash_state *S, uint32_t *hash) {
    S->buffer[60/4] = 0x01;
#pragma unroll 2
    for (int i=60/4+1; i < 72/4; ++i) S->buffer[i] = 0;
    S->buffer[72/4 - 1] |= 0x80000000;
    keccak_block(S, (const uint32_t*)S->buffer);
#pragma unroll 8
    for (size_t i = 0; i < 64; i += 8) {
        U64TO32_LE((&hash[i/4]), S->state[i / 8]);
    }
}

__device__ void
keccak_hash_finish64(keccak_hash_state *S, uint32_t *hash) {
    S->buffer[64/4] = 0x01;
#pragma unroll 1
    for (int i=64/4+1; i < 72/4; ++i) S->buffer[i] = 0;
    S->buffer[72/4 - 1] |= 0x80000000;
    keccak_block(S, (const uint32_t*)S->buffer);
#pragma unroll 8
    for (size_t i = 0; i < 64; i += 8) {
        U64TO32_LE((&hash[i/4]), S->state[i / 8]);
    }
}

// ---------------------------- END keccak functions ------------------------------------

// ---------------------------- BEGIN PBKDF2 functions ------------------------------------

typedef struct pbkdf2_hmac_state_t {
    keccak_hash_state inner, outer;
} pbkdf2_hmac_state;


__device__ void
pbkdf2_hash(uint32_t *hash, const uint32_t *m) {
    keccak_hash_state st;
    keccak_hash_init(&st);
    keccak_hash_update72(&st, m);
    keccak_hash_update8(&st, m+72/4);
    keccak_hash_finish8(&st, hash);
}

/* hmac */
__device__ void
pbkdf2_hmac_init80(pbkdf2_hmac_state *st, const uint32_t *key) {
    uint32_t pad[72/4];
    size_t i;

    keccak_hash_init(&st->inner);
    keccak_hash_init(&st->outer);

#pragma unroll 18
    for (i = 0; i < 72/4; i++)
        pad[i] = 0;

    /* key > blocksize bytes, hash it */
    pbkdf2_hash(pad, key);

    /* inner = (key ^ 0x36) */
    /* h(inner || ...) */
#pragma unroll 18
    for (i = 0; i < 72/4; i++)
        pad[i] ^= 0x36363636;
    keccak_hash_update72(&st->inner, pad);

    /* outer = (key ^ 0x5c) */
    /* h(outer || ...) */
#pragma unroll 18
    for (i = 0; i < 72/4; i++)
        pad[i] ^= 0x6a6a6a6a;
    keccak_hash_update72(&st->outer, pad);
}

// assuming there is no leftover data and exactly 72 bytes are incoming
// we can directly call into the block hashing function
__device__ void
pbkdf2_hmac_update72(pbkdf2_hmac_state *st, const uint32_t *m) {
    /* h(inner || m...) */
    keccak_hash_update72(&st->inner, m);
}

__device__ void
pbkdf2_hmac_update8(pbkdf2_hmac_state *st, const uint32_t *m) {
    /* h(inner || m...) */
    keccak_hash_update8(&st->inner, m);
}

__device__ void
pbkdf2_hmac_update4_8(pbkdf2_hmac_state *st, const uint32_t *m) {
    /* h(inner || m...) */
    keccak_hash_update4_8(&st->inner, m);
}

__device__ void
pbkdf2_hmac_update4_56(pbkdf2_hmac_state *st, const uint32_t *m) {
    /* h(inner || m...) */
    keccak_hash_update4_56(&st->inner, m);
}

__device__ void
pbkdf2_hmac_update56(pbkdf2_hmac_state *st, const uint32_t *m) {
    /* h(inner || m...) */
    keccak_hash_update56(&st->inner, m);
}

__device__ void
pbkdf2_hmac_finish12(pbkdf2_hmac_state *st, uint32_t *mac) {
    /* h(inner || m) */
    uint32_t innerhash[16];
    keccak_hash_finish12(&st->inner, innerhash);

    /* h(outer || h(inner || m)) */
    keccak_hash_update64(&st->outer, innerhash);
    keccak_hash_finish64(&st->outer, mac);
}

__device__ void
pbkdf2_hmac_finish60(pbkdf2_hmac_state *st, uint32_t *mac) {
    /* h(inner || m) */
    uint32_t innerhash[16];
    keccak_hash_finish60(&st->inner, innerhash);

    /* h(outer || h(inner || m)) */
    keccak_hash_update64(&st->outer, innerhash);
    keccak_hash_finish64(&st->outer, mac);
}

__device__ void
pbkdf2_statecopy8(pbkdf2_hmac_state *d, pbkdf2_hmac_state *s) {
    statecopy8(&d->inner, &s->inner);
    statecopy0(&d->outer, &s->outer);
}

// ---------------------------- END PBKDF2 functions ------------------------------------

static __device__ uint32_t cuda_swab32(uint32_t x)
{
    return (((x << 24) & 0xff000000u) | ((x << 8) & 0x00ff0000u)
          | ((x >> 8) & 0x0000ff00u) | ((x >> 24) & 0x000000ffu));
}

__global__ void cuda_pre_keccak512(uint32_t *g_idata, uint32_t nonce)
{
    nonce        +=       (blockIdx.x * blockDim.x) + threadIdx.x; 
    g_idata      += 32 * ((blockIdx.x * blockDim.x) + threadIdx.x);

    uint32_t data[20];

#pragma unroll 19
    for (int i=0; i <19; ++i)
        data[i] = cuda_swab32(pdata[i]);
    data[19] = cuda_swab32(nonce);

//    scrypt_pbkdf2_1((const uint8_t*)data, 80, (const uint8_t*)data, 80, (uint8_t*)g_idata, 128);

    pbkdf2_hmac_state hmac_pw, work;
    uint32_t ti[16];
    uint32_t be;
    
    /* hmac(password, ...) */
    pbkdf2_hmac_init80(&hmac_pw, data);

    /* hmac(password, salt...) */
    pbkdf2_hmac_update72(&hmac_pw, data);
    pbkdf2_hmac_update8(&hmac_pw, data+72/4);

    /* U1 = hmac(password, salt || be(i)) */
    be = cuda_swab32(1);
    pbkdf2_statecopy8(&work, &hmac_pw);
    pbkdf2_hmac_update4_8(&work, &be);
    pbkdf2_hmac_finish12(&work, ti);
    mycpy64(g_idata, ti);

    be = cuda_swab32(2);
    pbkdf2_statecopy8(&work, &hmac_pw);
    pbkdf2_hmac_update4_8(&work, &be);
    pbkdf2_hmac_finish12(&work, ti);
    mycpy64(g_idata+16, ti);
}


__global__ void cuda_post_keccak512(uint32_t *g_odata, uint32_t *g_hash, uint32_t nonce)
{
    nonce        +=       (blockIdx.x * blockDim.x) + threadIdx.x; 
    g_odata      += 32 * ((blockIdx.x * blockDim.x) + threadIdx.x);
    g_hash       +=  8 * ((blockIdx.x * blockDim.x) + threadIdx.x);

    uint32_t data[20];

#pragma unroll 19
    for (int i=0; i <19; ++i)
        data[i] = cuda_swab32(pdata[i]);
    data[19] = cuda_swab32(nonce);

//    scrypt_pbkdf2_1((const uint8_t*)data, 80, (const uint8_t*)g_odata, 128, (uint8_t*)g_hash, 32);

    pbkdf2_hmac_state hmac_pw;
    uint32_t ti[16];
    uint32_t be;
    
    /* hmac(password, ...) */
    pbkdf2_hmac_init80(&hmac_pw, data);

    /* hmac(password, salt...) */
    pbkdf2_hmac_update72(&hmac_pw, g_odata);
    pbkdf2_hmac_update56(&hmac_pw, g_odata+72/4);

    /* U1 = hmac(password, salt || be(i)) */
    be = cuda_swab32(1);
    pbkdf2_hmac_update4_56(&hmac_pw, &be);
    pbkdf2_hmac_finish60(&hmac_pw, ti);
    mycpy32(g_hash, ti);
}

//
// callable host code to initialize constants and to call kernels
//

extern "C" void prepare_keccak512(int thr_id, uint32_t host_pdata[20])
{
    static bool init[8] = {false, false, false, false, false, false, false, false};
    if (!init[thr_id])
    {
        cudaMemcpyToSymbol(c_keccak_round_constants, host_keccak_round_constants, sizeof(host_keccak_round_constants), 0, cudaMemcpyHostToDevice);
        init[thr_id] = true;
    }
    cudaMemcpyToSymbol(pdata, host_pdata, 20*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

extern "C" void pre_keccak512(int thr_id, int stream, uint32_t nonce, int throughput)
{
    dim3 block(128);
    dim3 grid((throughput+127)/128);

    cuda_pre_keccak512<<<grid, block, 0, context_streams[stream][thr_id]>>>(context_idata[stream][thr_id], nonce);
}

extern "C" void post_keccak512(int thr_id, int stream, uint32_t nonce, uint32_t hash[8], int throughput)
{
    dim3 block(128);
    dim3 grid((throughput+127)/128);

    cuda_post_keccak512<<<grid, block, 0, context_streams[stream][thr_id]>>>(context_odata[stream][thr_id], context_hash[stream][thr_id], nonce);

    unsigned int mem_size = throughput * sizeof(uint32_t) * 8;

    // copy device memory to host
    checkCudaErrors(cudaMemcpyAsync(hash, context_hash[stream][thr_id], mem_size,
                    cudaMemcpyDeviceToHost, context_streams[stream][thr_id]));
}


//
// Maxcoin related Keccak implementation (Keccak256)
//

#include <stdint.h>

#define ROL(a, offset) ((((uint64_t)a) << ((offset) % 64)) ^ (((uint64_t)a) >> (64-((offset) % 64))))
#define ROL_mult8(a, offset) ROL(a, offset)

__constant__ uint64_t KeccakF_RoundConstants[24];

uint64_t host_KeccakF_RoundConstants[24] = 
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

__device__ __forceinline__ void KeccakF( uint64_t *state, const uint64_t *in )
{
#pragma unroll 17
    for (int laneCount = 0; laneCount < 17; ++laneCount)
        state[laneCount] ^= in[laneCount];

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
        Aba = state[ 0];
        Abe = state[ 1];
        Abi = state[ 2];
        Abo = state[ 3];
        Abu = state[ 4];
        Aga = state[ 5];
        Age = state[ 6];
        Agi = state[ 7];
        Ago = state[ 8];
        Agu = state[ 9];
        Aka = state[10];
        Ake = state[11];
        Aki = state[12];
        Ako = state[13];
        Aku = state[14];
        Ama = state[15];
        Ame = state[16];
        Ami = state[17];
        Amo = state[18];
        Amu = state[19];
        Asa = state[20];
        Ase = state[21];
        Asi = state[22];
        Aso = state[23];
        Asu = state[24];

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

        //copyToState(state, A)
        state[ 0] = Aba;
        state[ 1] = Abe;
        state[ 2] = Abi;
        state[ 3] = Abo;
        state[ 4] = Abu;
        state[ 5] = Aga;
        state[ 6] = Age;
        state[ 7] = Agi;
        state[ 8] = Ago;
        state[ 9] = Agu;
        state[10] = Aka;
        state[11] = Ake;
        state[12] = Aki;
        state[13] = Ako;
        state[14] = Aku;
        state[15] = Ama;
        state[16] = Ame;
        state[17] = Ami;
        state[18] = Amo;
        state[19] = Amu;
        state[20] = Asa;
        state[21] = Ase;
        state[22] = Asi;
        state[23] = Aso;
        state[24] = Asu;
    }
}

__constant__ uint64_t pdata64[10];

__global__ void crypto_hash( uint64_t *out, uint32_t nonce )
{
    out += 4 * ((blockIdx.x * blockDim.x) + threadIdx.x);
    nonce = cuda_swab32(nonce + ((blockIdx.x * blockDim.x) + threadIdx.x));

    uint64_t temp[17]; // 136 bytes

    // padding
#pragma unroll 10
    for (int i=0; i < 9;  ++i) temp[i]  = pdata64[i];
    // mask out nonce from pdata64 and insert the thread specific nonce
    temp[9]  = (pdata64[9] & 0x00000000FFFFFFFFULL) | (((uint64_t)nonce) << 32);
    // padding
    temp[10] = 0x0000000000000001ULL;
    temp[12] = 0;
    temp[13] = 0;
    temp[14] = 0;
    temp[15] = 0;
    temp[16] = 0x8000000000000000ULL;

    uint64_t state[25];
#pragma unroll 25
    for (int i=0; i < 25; ++i) state[i] = 0;
    KeccakF( state, (const uint64_t*)temp );

#pragma unroll 4
    for (int i=0; i < 4; ++i) out[i] = state[i];
}


extern "C" void prepare_keccak256(int thr_id, uint32_t host_pdata[20])
{
    static bool init[8] = {false, false, false, false, false, false, false, false};
    if (!init[thr_id])
    {
        cudaMemcpyToSymbol(KeccakF_RoundConstants, host_KeccakF_RoundConstants, sizeof(host_KeccakF_RoundConstants), 0, cudaMemcpyHostToDevice);
        init[thr_id] = true;
    }
    cudaMemcpyToSymbol(pdata64, host_pdata, 20*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

#include <map>
extern std::map<int, int> context_blocks;
extern std::map<int, int> context_wpb;
extern std::map<int, KernelInterface *> context_kernel;

extern "C" void do_keccak256(int thr_id, int stream, uint32_t *hash, uint32_t nonce, int throughput)
{
    unsigned int GRID_BLOCKS = context_blocks[thr_id];
    unsigned int WARPS_PER_BLOCK = context_wpb[thr_id];
    unsigned int THREADS_PER_WU = context_kernel[thr_id]->threads_per_wu();

    // setup execution parameters
    dim3  grid(WU_PER_LAUNCH/WU_PER_BLOCK, 1, 1);
    dim3  threads(THREADS_PER_WU*WU_PER_BLOCK, 1, 1);

    crypto_hash<<<grid, threads, 0, context_streams[stream][thr_id]>>>((uint64_t*)context_hash[stream][thr_id], nonce);

    size_t mem_size = throughput * sizeof(uint32_t) * 8;

    // copy device memory to host
    checkCudaErrors(cudaMemcpyAsync(hash, context_hash[stream][thr_id], mem_size,
                    cudaMemcpyDeviceToHost, context_streams[stream][thr_id]));
}
