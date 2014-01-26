//
//  =============== KECCAK part on nVidia GPU ======================
//
// NOTE: compile this .cu module for compute_10,sm_10 with --maxrregcount=64
//
// TODO: the actual CUDA porting work is work in progress...
//
//       For good performance we have to get rid of most local memory spills
//       TODO: drop the uint8_t conversions (use uint32_t arrays or uint64_t)
//             manually inline scrypt_pbkdf2_1 function into pre/post kernels
//             and unroll loops. Remove unused code.

#include <map>
#include <stdint.h>

#include "salsa_kernel.h"
#include "miner.h"

#include "keccak.h"

// define some error checking macros
#undef checkCudaErrors

#define checkCudaErrors(x) \
{ \
    cudaGetLastError(); \
    x; \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
    { \
        applog(LOG_ERR, "GPU #%d: cudaError %d (%s) calling '%s' (%s line %d)\n", device_map[thr_id], err, cudaGetErrorString(err), #x, __FILE__, __LINE__); \
    } \
}

// from salsa_kernel.cu
extern std::map<int, uint32_t *> context_idata[2];
extern std::map<int, uint32_t *> context_odata[2];
extern std::map<int, cudaStream_t> context_streams[2];
extern std::map<int, uint32_t *> context_hash[2];

#define U8TO32_BE(p)                                            \
	(((uint32_t)((p)[0]) << 24) | ((uint32_t)((p)[1]) << 16) |  \
	 ((uint32_t)((p)[2]) <<  8) | ((uint32_t)((p)[3])      ))

#define U8TO32_LE(p)                                            \
	(((uint32_t)((p)[0])      ) | ((uint32_t)((p)[1]) <<  8) |  \
	 ((uint32_t)((p)[2]) << 16) | ((uint32_t)((p)[3]) << 24))

#define U32TO8_BE(p, v)                                           \
	(p)[0] = (uint8_t)((v) >> 24); (p)[1] = (uint8_t)((v) >> 16); \
	(p)[2] = (uint8_t)((v) >>  8); (p)[3] = (uint8_t)((v)      );

#define U32TO8_LE(p, v)                                           \
	(p)[0] = (uint8_t)((v)      ); (p)[1] = (uint8_t)((v) >>  8); \
	(p)[2] = (uint8_t)((v) >> 16); (p)[3] = (uint8_t)((v) >> 24);

#define U8TO64_BE(p)                                                  \
	(((uint64_t)U8TO32_BE(p) << 32) | (uint64_t)U8TO32_BE((p) + 4))

#define U8TO64_LE(p)                                                  \
	(((uint64_t)U8TO32_LE(p)) | ((uint64_t)U8TO32_LE((p) + 4) << 32))

#define U64TO8_BE(p, v)                        \
	U32TO8_BE((p),     (uint32_t)((v) >> 32)); \
	U32TO8_BE((p) + 4, (uint32_t)((v)      ));

#define U64TO8_LE(p, v)                        \
	U32TO8_LE((p),     (uint32_t)((v)      )); \
	U32TO8_LE((p) + 4, (uint32_t)((v) >> 32));

#define U32_SWAP(v) {                                             \
	(v) = (((v) << 8) & 0xFF00FF00 ) | (((v) >> 8) & 0xFF00FF );  \
    (v) = ((v) << 16) | ((v) >> 16);                              \
}

#define U64_SWAP(v) {                                                                       \
	(v) = (((v) <<  8) & 0xFF00FF00FF00FF00ull ) | (((v) >>  8) & 0x00FF00FF00FF00FFull );  \
	(v) = (((v) << 16) & 0xFFFF0000FFFF0000ull ) | (((v) >> 16) & 0x0000FFFF0000FFFFull );  \
    (v) = ((v) << 32) | ((v) >> 32);                                                        \
}

#define ROTL64(a,b) (((a) << (b)) | ((a) >> (64 - b)))

// ---------------------------- BEGIN keccak functions ------------------------------------

#define KECCAK_HASH "Keccak-512"
#define KECCAK_HASH_DIGEST_SIZE 64
#define KECCAK_F 1600
#define KECCAK_C (KECCAK_HASH_DIGEST_SIZE * 8 * 2) /* 1024 */
#define KECCAK_R (KECCAK_F - KECCAK_C) /* 576 */
#define KECCAK_HASH_BLOCK_SIZE (KECCAK_R / 8)

typedef uint8_t keccak_hash_digest[KECCAK_HASH_DIGEST_SIZE];

typedef struct keccak_hash_state_t {
	uint64_t state[KECCAK_F / 64]; // 25
	uint32_t leftover;
	uint8_t buffer[KECCAK_HASH_BLOCK_SIZE]; // 72
} keccak_hash_state;

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
keccak_block(keccak_hash_state *S, const uint8_t *in) {
	size_t i;
	uint64_t *s = S->state, t[5], u[5], v, w;

	/* absorb input */
	for (i = 0; i < KECCAK_HASH_BLOCK_SIZE / 8; i++, in += 8)
		s[i] ^= U8TO64_LE(in);
	
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
	memset(S, 0, sizeof(*S));
}

__device__ void
keccak_hash_update(keccak_hash_state *S, const uint8_t *in, size_t inlen) {
	size_t want;

	/* handle the previous data */
	if (S->leftover) {
		want = (KECCAK_HASH_BLOCK_SIZE - S->leftover);
		want = (want < inlen) ? want : inlen;
		memcpy(S->buffer + S->leftover, in, want);
		S->leftover += (uint32_t)want;
		if (S->leftover < KECCAK_HASH_BLOCK_SIZE)
			return;
		in += want;
		inlen -= want;
		keccak_block(S, S->buffer);
	}

	/* handle the current data */
	while (inlen >= KECCAK_HASH_BLOCK_SIZE) {
		keccak_block(S, in);
		in += KECCAK_HASH_BLOCK_SIZE;
		inlen -= KECCAK_HASH_BLOCK_SIZE;
	}

	/* handle leftover data */
	S->leftover = (uint32_t)inlen;
	if (S->leftover)
		memcpy(S->buffer, in, S->leftover);
}

__device__ void
keccak_hash_finish(keccak_hash_state *S, uint8_t *hash) {
	size_t i;

	S->buffer[S->leftover] = 0x01;
	memset(S->buffer + (S->leftover + 1), 0, KECCAK_HASH_BLOCK_SIZE - (S->leftover + 1));
	S->buffer[KECCAK_HASH_BLOCK_SIZE - 1] |= 0x80;
	keccak_block(S, S->buffer);

	for (i = 0; i < KECCAK_HASH_DIGEST_SIZE; i += 8) {
		U64TO8_LE(&hash[i], S->state[i / 8]);
	}
}

// ---------------------------- END keccak functions ------------------------------------

// ---------------------------- BEGIN PBKDF2 functions ------------------------------------

typedef struct scrypt_hmac_state_t {
	keccak_hash_state inner, outer;
} scrypt_hmac_state;


__device__ void
scrypt_hash(keccak_hash_digest hash, const uint8_t *m, size_t mlen) {
	keccak_hash_state st;
	keccak_hash_init(&st);
	keccak_hash_update(&st, m, mlen);
	keccak_hash_finish(&st, hash);
}

/* hmac */
__device__ void
scrypt_hmac_init(scrypt_hmac_state *st, const uint8_t *key, size_t keylen) {
	uint8_t pad[KECCAK_HASH_BLOCK_SIZE] = {0};
	size_t i;

	keccak_hash_init(&st->inner);
	keccak_hash_init(&st->outer);

	if (keylen <= KECCAK_HASH_BLOCK_SIZE) {
		/* use the key directly if it's <= blocksize bytes */
		memcpy(pad, key, keylen);
	} else {
		/* if it's > blocksize bytes, hash it */
		scrypt_hash(pad, key, keylen);
	}

	/* inner = (key ^ 0x36) */
	/* h(inner || ...) */
	for (i = 0; i < KECCAK_HASH_BLOCK_SIZE; i++)
		pad[i] ^= 0x36;
	keccak_hash_update(&st->inner, pad, KECCAK_HASH_BLOCK_SIZE);

	/* outer = (key ^ 0x5c) */
	/* h(outer || ...) */
	for (i = 0; i < KECCAK_HASH_BLOCK_SIZE; i++)
		pad[i] ^= (0x5c ^ 0x36);
	keccak_hash_update(&st->outer, pad, KECCAK_HASH_BLOCK_SIZE);
}

__device__ void
scrypt_hmac_update(scrypt_hmac_state *st, const uint8_t *m, size_t mlen) {
	/* h(inner || m...) */
	keccak_hash_update(&st->inner, m, mlen);
}

__device__ void
scrypt_hmac_finish(scrypt_hmac_state *st, keccak_hash_digest mac) {
	/* h(inner || m) */
	keccak_hash_digest innerhash;
	keccak_hash_finish(&st->inner, innerhash);

	/* h(outer || h(inner || m)) */
	keccak_hash_update(&st->outer, innerhash, sizeof(innerhash));
	keccak_hash_finish(&st->outer, mac);
}

/*
 * Special version where N = 1
 *  - mikaelh
 */
__device__ void
scrypt_pbkdf2_1(const uint8_t *password, size_t password_len, const uint8_t *salt, size_t salt_len, uint8_t *out, size_t bytes) {
	scrypt_hmac_state hmac_pw, hmac_pw_salt, work;
	keccak_hash_digest ti, u;
	uint8_t be[4];
	uint32_t i, /*j,*/ blocks;
//	uint64_t c;
	
	/* bytes must be <= (0xffffffff - (SCRYPT_HASH_DIGEST_SIZE - 1)), which they will always be under scrypt */

	/* hmac(password, ...) */
	scrypt_hmac_init(&hmac_pw, password, password_len);

	/* hmac(password, salt...) */
	hmac_pw_salt = hmac_pw;
	scrypt_hmac_update(&hmac_pw_salt, salt, salt_len);

	blocks = ((uint32_t)bytes + (KECCAK_HASH_DIGEST_SIZE - 1)) / KECCAK_HASH_DIGEST_SIZE;
	for (i = 1; i <= blocks; i++) {
		/* U1 = hmac(password, salt || be(i)) */
		U32TO8_BE(be, i);
		work = hmac_pw_salt;
		scrypt_hmac_update(&work, be, 4);
		scrypt_hmac_finish(&work, ti);
		memcpy(u, ti, sizeof(u));

		memcpy(out, ti, (bytes > KECCAK_HASH_DIGEST_SIZE) ? KECCAK_HASH_DIGEST_SIZE : bytes);
		out += KECCAK_HASH_DIGEST_SIZE;
		bytes -= KECCAK_HASH_DIGEST_SIZE;
	}
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

    scrypt_pbkdf2_1((const uint8_t*)data, 80, (const uint8_t*)data, 80, (uint8_t*)g_idata, 128);
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

    scrypt_pbkdf2_1((const uint8_t*)data, 80, (const uint8_t*)g_odata, 128, (uint8_t*)g_hash, 32);
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
