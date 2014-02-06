#include "cpuminer-config.h"
#include "miner.h"
#include "salsa_kernel.h"

#include <string.h>
#include <stdint.h>

// an alternative SHA-3 implementation, easier portable to CUDA IMHO
#include "sha3.h"
#include "keccak.h"

int scanhash_keccak(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, struct timeval *tv_start, struct timeval *tv_end, unsigned long *hashes_done)
{
	int throughput = cuda_throughput(thr_id);

	gettimeofday(tv_start, NULL);

	uint32_t n = pdata[19] - 1;
	const uint32_t first_nonce = pdata[19];
	const uint32_t Htarg = ptarget[7];

	uint32_t endiandata[20];
	for (int kk=0; kk < 20; kk++)
		be32enc(&endiandata[kk], ((uint32_t*)pdata)[kk]);

	prepare_keccak256(thr_id, endiandata);

	uint32_t *cuda_hash64 = (uint32_t *)cuda_hashbuffer(thr_id, 0);

	do {
		int nonce = n+1;

		do_keccak256(thr_id, 0, cuda_hash64, nonce, throughput);
		n += throughput;

		cuda_scrypt_sync(thr_id, 0);

		// CPU validation
#if 0
		for (int i=0; i < throughput; ++i)
		{
			uint32_t hash64[8];
			be32enc(&endiandata[19], nonce+i); 
			crypto_hash( (unsigned char*)hash64, (unsigned char*)&endiandata[0], 80 );

			if (memcmp(hash64, &cuda_hash64[8*i], 32))
				fprintf(stderr, "CPU and CUDA hashes (i=%d) differ!\n", i);
		}
#endif

		for (int i=0; i < throughput; ++i)
		{
			uint32_t *hash64 = &cuda_hash64[8*i];
			if (hash64[7] <= Htarg &&
					fulltest(hash64, ptarget)) {
				pdata[19] = nonce+i;
				*hashes_done = n - first_nonce + 1;
				gettimeofday(tv_end, NULL);
				return true;
			}
		}
	} while (n < max_nonce && !work_restart[thr_id].restart);
	
	*hashes_done = n - first_nonce + 1;
	pdata[19] = n;
	gettimeofday(tv_end, NULL);
	return 0;
}
