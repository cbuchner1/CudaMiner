#include "cpuminer-config.h"
#include "miner.h"
#include "salsa_kernel.h"

#include <string.h>
#include <stdint.h>

#include "blake.h"

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

const uint8_t sigma[][16] = {
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

const uint32_t cst[16] = {
  0x243F6A88,0x85A308D3,0x13198A2E,0x03707344,
  0xA4093822,0x299F31D0,0x082EFA98,0xEC4E6C89,
  0x452821E6,0x38D01377,0xBE5466CF,0x34E90C6C,
  0xC0AC29B7,0xC97C50DD,0x3F84D5B5,0xB5470917};

const uint8_t padding[] =
  {0x80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


void blake256_compress( state *S, const uint8_t *block ) {

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


void blake256_init( state *S ) {

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


void blake256_update( state *S, const uint8_t *data, size_t datalen ) {

  int left=S->buflen >> 3; 
  int fill=64 - left;
    
  if( left && ( ((datalen >> 3) & 0x3F) >= (unsigned)fill ) ) {
    memcpy( (void*) (S->buf + left), (void*) data, fill );
    S->t[0] += 512;
    if (S->t[0] == 0) S->t[1]++;      
    blake256_compress( S, S->buf );
    data += fill;
    datalen  -= (fill << 3);       
    left = 0;
  }

  while( datalen >= 512 ) {
    S->t[0] += 512;
    if (S->t[0] == 0) S->t[1]++;
    blake256_compress( S, data );
    data += 64;
    datalen  -= 512;
  }
  
  if( datalen > 0 ) {
    memcpy( (void*) (S->buf + left), (void*) data, datalen>>3 );
    S->buflen = (left<<3) + datalen;
  }
  else S->buflen=0;
}


void blake256_final( state *S, uint8_t *digest ) {
  
  uint8_t msglen[8], zo=0x01, oo=0x81;
  uint32_t lo=S->t[0] + S->buflen, hi=S->t[1];
  if ( lo < (unsigned)S->buflen ) hi++;
  U32TO8(  msglen + 0, hi );
  U32TO8(  msglen + 4, lo );

  if ( S->buflen == 440 ) { /* one padding byte */
    S->t[0] -= 8;
    blake256_update( S, &oo, 8 );
  }
  else {
    if ( S->buflen < 440 ) { /* enough space to fill the block  */
      if ( !S->buflen ) S->nullt=1;
      S->t[0] -= 440 - S->buflen;
      blake256_update( S, padding, 440 - S->buflen );
    }
    else { /* need 2 compressions */
      S->t[0] -= 512 - S->buflen; 
      blake256_update( S, padding, 512 - S->buflen );
      S->t[0] -= 440;
      blake256_update( S, padding+1, 440 );
      S->nullt = 1;
    }
    blake256_update( S, &zo, 8 );
    S->t[0] -= 8;
  }
  S->t[0] -= 64;
  blake256_update( S, msglen, 64 );    
  
  U32TO8( digest + 0, S->h[0]);
  U32TO8( digest + 4, S->h[1]);
  U32TO8( digest + 8, S->h[2]);
  U32TO8( digest +12, S->h[3]);
  U32TO8( digest +16, S->h[4]);
  U32TO8( digest +20, S->h[5]);
  U32TO8( digest +24, S->h[6]);
  U32TO8( digest +28, S->h[7]);
}


void blake256_hash( uint8_t *out, const uint8_t *in, size_t inlen ) {

  state S;  
  blake256_init( &S );
  blake256_update( &S, in, inlen*8 );
  blake256_final( &S, out );
}

int scanhash_blake(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, struct timeval *tv_start, struct timeval *tv_end, unsigned long *hashes_done)
{
	int throughput = cuda_throughput(thr_id);

	gettimeofday(tv_start, NULL);

	uint32_t n = pdata[19] - 1;
	
	// TESTING ONLY
//	((uint32_t*)ptarget)[7] = 0x0000000f;
	
	const uint32_t Htarg = ptarget[7];

	uint32_t endiandata[20];
	for (int kk=0; kk < 20; kk++)
		be32enc(&endiandata[kk], pdata[kk]);

	cuda_prepare_blake256(thr_id, endiandata, ptarget);

	uint32_t *cuda_hash64[2] = { (uint32_t *)cuda_hashbuffer(thr_id, 0), (uint32_t *)cuda_hashbuffer(thr_id, 1) };
	memset(cuda_hash64[0], 0xff, throughput * 8 * sizeof(uint32_t));
	memset(cuda_hash64[1], 0xff, throughput * 8 * sizeof(uint32_t));

	bool validate = false;
	uint32_t nonce[2];
	int cur = 0, nxt = 1;

	// begin work on first CUDA stream
	nonce[cur] = n+1; n += throughput;
	cuda_do_blake256(thr_id, 0, cuda_hash64[cur], nonce[cur], throughput, validate);

	do {

		nonce[nxt] = n+1; n += throughput;
		if ((n-throughput) < max_nonce && !work_restart[thr_id].restart)
		{
			// begin work on next CUDA stream
			cuda_do_blake256(thr_id, 0, cuda_hash64[nxt], nonce[nxt], throughput, validate);
		}

		// synchronize current stream and get the "winning" nonce index, if any
		cuda_scrypt_sync(thr_id, cur);
		uint32_t result =  *cuda_hash64[cur];

		// optional full CPU based validation (see validate flag)
		if (validate)
		{
			for (int i=0; i < throughput; ++i)
			{
				uint32_t hash64[8];
				be32enc(&endiandata[19], nonce[cur]+i); 
				blake256_hash( (unsigned char*)hash64, (unsigned char*)&endiandata[0], 80 );
	
				if (memcmp(hash64, &cuda_hash64[cur][8*i], 32))
					fprintf(stderr, "CPU and CUDA hashes (i=%d) differ!\n", i);
			}
		}
		else if (result != 0xffffffff && result > pdata[19])
		{
			uint32_t hash64[8];
			be32enc(&endiandata[19], result);
			blake256_hash( (unsigned char*)hash64, (unsigned char*)&endiandata[0], 80 );
			if (result >= nonce[cur] && result < nonce[cur]+throughput && hash64[7] <= Htarg && fulltest(hash64, ptarget)) {
				*hashes_done = n-throughput - pdata[19] + 1;
				pdata[19] = result;
				gettimeofday(tv_end, NULL);
				return true;
			} else {
				applog(LOG_INFO, "GPU #%d: %s result for nonce $%08x does not validate on CPU!", device_map[thr_id], device_name[thr_id], result);
			}
		}
		cur = (cur + 1) % 2;
		nxt = (nxt + 1) % 2;
	} while ((n-throughput) < max_nonce && !work_restart[thr_id].restart);
	
	*hashes_done = n-throughput - pdata[19] + 1;
	if (n-throughput > pdata[19])
		// CB: don't report values bigger max_nonce
		pdata[19] = max_nonce > n-throughput ? n-throughput : max_nonce;
	else
		pdata[19] = 0xffffffffU; // CB: prevent nonce space overflow.
	gettimeofday(tv_end, NULL);
	return 0;
}
