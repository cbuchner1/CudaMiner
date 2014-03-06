//
//  =============== BLAKE part on nVidia GPU ======================
//
// This is the generic "default" implementation when no architecture
// specific implementation is available in the kernel.
//
// NOTE: compile this .cu module for compute_10,sm_10 with --maxrregcount=64
//
// TODO: CUDA porting work remains to be done.
//

#include <map>
#include <stdint.h>

#include "salsa_kernel.h"
#include "miner.h"

#include "blake.h"

__constant__ uint64_t ptarget64[4];
__constant__ uint32_t pdata[20];

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
extern std::map<int, uint32_t *> context_hash[2];

#define U8TO32(p)					\
  (((uint32_t)((p)[0]) << 24) | ((uint32_t)((p)[1]) << 16) |	\
   ((uint32_t)((p)[2]) <<  8) | ((uint32_t)((p)[3])      ))
#define U32TO8(p, v)					\
  (p)[0] = (uint8_t)((v) >> 24); (p)[1] = (uint8_t)((v) >> 16);	\
  (p)[2] = (uint8_t)((v) >>  8); (p)[3] = (uint8_t)((v)      ); 

typedef struct  { 
  uint32_t h[8], s[4], t[2];
  int buflen, nullt;
  uint8_t  buf[64];
} state;

__constant__ uint8_t sigma[14][16];

const uint8_t host_sigma[14][16] = {
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
  {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 },
  {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4 },
  { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8 },
  { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13 },
  { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9 },
  {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11 },
  {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10 },
  { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5 },
  {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13 ,0 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
  {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 },
  {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4 },
  { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8 }};

__constant__ uint32_t cst[16];

const uint32_t host_cst[16] = {
  0x243F6A88,0x85A308D3,0x13198A2E,0x03707344,
  0xA4093822,0x299F31D0,0x082EFA98,0xEC4E6C89,
  0x452821E6,0x38D01377,0xBE5466CF,0x34E90C6C,
  0xC0AC29B7,0xC97C50DD,0x3F84D5B5,0xB5470917};

__constant__ uint8_t padding[64];

const uint8_t host_padding[64] =
  {0x80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


__device__ void cuda_blake256_compress( state *S, const uint8_t *block ) {

  uint32_t v[16], m[16], i;
#define ROT(x,n) (((x)<<(32-n))|( (x)>>(n)))
#define G(a,b,c,d,e)					\
  v[a] += (m[sigma[i][e]] ^ cst[sigma[i][e+1]]) + v[b];	\
  v[d] = ROT( v[d] ^ v[a],16);				\
  v[c] += v[d];						\
  v[b] = ROT( v[b] ^ v[c],12);				\
  v[a] += (m[sigma[i][e+1]] ^ cst[sigma[i][e]])+v[b];	\
  v[d] = ROT( v[d] ^ v[a], 8);				\
  v[c] += v[d];						\
  v[b] = ROT( v[b] ^ v[c], 7);				
							
  for(i=0; i<16;++i)  m[i] = U8TO32(block + i*4);
  for(i=0; i< 8;++i)  v[i] = S->h[i];
  v[ 8] = S->s[0] ^ 0x243F6A88;
  v[ 9] = S->s[1] ^ 0x85A308D3;
  v[10] = S->s[2] ^ 0x13198A2E;
  v[11] = S->s[3] ^ 0x03707344;
  v[12] =  0xA4093822;
  v[13] =  0x299F31D0;
  v[14] =  0x082EFA98;
  v[15] =  0xEC4E6C89;
  if (S->nullt == 0) { 
    v[12] ^= S->t[0];
    v[13] ^= S->t[0];
    v[14] ^= S->t[1];
    v[15] ^= S->t[1];
  }

  for(i=0; i<14; ++i) {
    G( 0, 4, 8,12, 0);
    G( 1, 5, 9,13, 2);
    G( 2, 6,10,14, 4);
    G( 3, 7,11,15, 6);
    G( 3, 4, 9,14,14);   
    G( 2, 7, 8,13,12);
    G( 0, 5,10,15, 8);
    G( 1, 6,11,12,10);
  }
  
  for(i=0; i<16;++i)  S->h[i%8] ^= v[i]; 
  for(i=0; i<8 ;++i)  S->h[i] ^= S->s[i%4]; 
}


__device__ void cuda_blake256_init( state *S ) {

  S->h[0]=0x6A09E667;
  S->h[1]=0xBB67AE85;
  S->h[2]=0x3C6EF372;
  S->h[3]=0xA54FF53A;
  S->h[4]=0x510E527F;
  S->h[5]=0x9B05688C;
  S->h[6]=0x1F83D9AB;
  S->h[7]=0x5BE0CD19;
  S->t[0]=S->t[1]=S->buflen=S->nullt=0;
  S->s[0]=S->s[1]=S->s[2]=S->s[3] =0;
}


__device__ void cuda_blake256_update( state *S, const uint8_t *data, size_t datalen ) {

  int left=S->buflen >> 3; 
  int fill=64 - left;
    
  if( left && ( ((datalen >> 3) & 0x3F) >= (unsigned)fill ) ) {
    memcpy( (void*) (S->buf + left), (void*) data, fill );
    S->t[0] += 512;
    if (S->t[0] == 0) S->t[1]++;      
    cuda_blake256_compress( S, S->buf );
    data += fill;
    datalen  -= (fill << 3);       
    left = 0;
  }

  while( datalen >= 512 ) {
    S->t[0] += 512;
    if (S->t[0] == 0) S->t[1]++;
    cuda_blake256_compress( S, data );
    data += 64;
    datalen  -= 512;
  }
  
  if( datalen > 0 ) {
    memcpy( (void*) (S->buf + left), (void*) data, datalen>>3 );
    S->buflen = (left<<3) + datalen;
  }
  else S->buflen=0;
}


__device__ void cuda_blake256_final( state *S, uint8_t *digest ) {
  
  uint8_t msglen[8], zo=0x01, oo=0x81;
  uint32_t lo=S->t[0] + S->buflen, hi=S->t[1];
  if ( lo < (unsigned)S->buflen ) hi++;
  U32TO8(  msglen + 0, hi );
  U32TO8(  msglen + 4, lo );

  if ( S->buflen == 440 ) { /* one padding byte */
    S->t[0] -= 8;
    cuda_blake256_update( S, &oo, 8 );
  }
  else {
    if ( S->buflen < 440 ) { /* enough space to fill the block  */
      if ( !S->buflen ) S->nullt=1;
      S->t[0] -= 440 - S->buflen;
      cuda_blake256_update( S, padding, 440 - S->buflen );
    }
    else { /* need 2 compressions */
      S->t[0] -= 512 - S->buflen; 
      cuda_blake256_update( S, padding, 512 - S->buflen );
      S->t[0] -= 440;
      cuda_blake256_update( S, padding+1, 440 );
      S->nullt = 1;
    }
    cuda_blake256_update( S, &zo, 8 );
    S->t[0] -= 8;
  }
  S->t[0] -= 64;
  cuda_blake256_update( S, msglen, 64 );    
  
  U32TO8( digest + 0, S->h[0]);
  U32TO8( digest + 4, S->h[1]);
  U32TO8( digest + 8, S->h[2]);
  U32TO8( digest +12, S->h[3]);
  U32TO8( digest +16, S->h[4]);
  U32TO8( digest +20, S->h[5]);
  U32TO8( digest +24, S->h[6]);
  U32TO8( digest +28, S->h[7]);
}


__device__ void cuda_blake256_hash( uint8_t *out, const uint8_t *in, size_t inlen ) {

  state S;  
  cuda_blake256_init( &S );
  cuda_blake256_update( &S, in, inlen*8 );
  cuda_blake256_final( &S, out );
}


static __device__ uint32_t cuda_swab32(uint32_t x)
{
    return (((x << 24) & 0xff000000u) | ((x << 8) & 0x00ff0000u)
          | ((x >> 8) & 0x0000ff00u) | ((x >> 24) & 0x000000ffu));
}

__global__ void cuda_blake256_hash( uint64_t *g_out, uint32_t nonce, uint32_t *g_good, bool validate )
{
    uint32_t data[20];
    uint64_t out[4];

#pragma unroll 19
    for (int i=0; i < 19; ++i) data[i] = pdata[i];
    data[19] = cuda_swab32(nonce + ((blockIdx.x * blockDim.x) + threadIdx.x));

    state S;
    cuda_blake256_init( &S );
    cuda_blake256_update( &S, (const uint8_t*)data, 80*8 );
    cuda_blake256_final( &S, (uint8_t*)out );

    if (validate)
    {
        g_out += 4 * ((blockIdx.x * blockDim.x) + threadIdx.x);
#pragma unroll 4
        for (int i=0; i < 4; ++i) g_out[i] = out[i];
    }

    uint64_t *g_good64 = (uint64_t*)g_good;
    if (out[3] <=  ptarget64[3]) {
        if (out[3] < g_good64[3]) {
            g_good64[3] = out[3];
            g_good64[2] = out[2];
            g_good64[1] = out[1];
            g_good64[0] = out[0];
            g_good[8] = nonce + ((blockIdx.x * blockDim.x) + threadIdx.x);
        }
    }
}

static std::map<int, uint32_t *> context_good[2];

extern "C" void default_prepare_blake256(int thr_id, const uint32_t host_pdata[20], const uint32_t host_ptarget[8])
{
    static bool init[8] = {false, false, false, false, false, false, false, false};
    if (!init[thr_id])
    {
        checkCudaErrors(cudaMemcpyToSymbol(sigma, host_sigma, sizeof(host_sigma), 0, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(cst, host_cst, sizeof(host_cst), 0, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyToSymbol(padding, host_padding, sizeof(host_padding), 0, cudaMemcpyHostToDevice));

        // allocate pinned host memory for good hashes
        uint32_t *tmp;
        checkCudaErrors(cudaMalloc((void **) &tmp, 9*sizeof(uint32_t))); context_good[0][thr_id] = tmp;
        checkCudaErrors(cudaMalloc((void **) &tmp, 9*sizeof(uint32_t))); context_good[1][thr_id] = tmp;

        init[thr_id] = true;
    }
    checkCudaErrors(cudaMemcpyToSymbol(pdata, host_pdata, 20*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(ptarget64, host_ptarget, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}

extern "C" bool default_do_blake256(dim3 grid, dim3 threads, int thr_id, int stream, uint32_t *hash, uint32_t nonce, int throughput, bool do_d2h)
{
    bool success = true;
  
    checkCudaErrors(cudaMemsetAsync(context_good[stream][thr_id], 0xff, 9 * sizeof(uint32_t), context_streams[stream][thr_id]));

    cuda_blake256_hash<<<grid, threads, 0, context_streams[stream][thr_id]>>>((uint64_t*)context_hash[stream][thr_id], nonce, context_good[stream][thr_id], do_d2h);

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
